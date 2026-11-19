"""IP-20a: the pure derived Wake view (IP-20 D1 r3 §4-2, §9-1).

Every state of the table at its boundaries (half-open windows), the state
precedence, ``EXPIRED_PENDING_LEASE`` derived with no row, the A-4 terminal
consistency check, and the purity of the module.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect

import pytest

from suzka.r16.cycle_contracts import CycleClosureV1, LogicalCycleResolutionV1
from suzka.r16.history_reader import AdoptedRecord
from suzka.r16.history_records import HistoryOperationKind, RecordKind
from suzka.r19 import ledger
from suzka.r19.contracts import (
    AttemptOutcome,
    RefuseReason,
    TerminalReason,
    WakeAttemptOutcomeV1,
    WakeClaimV1,
    WakeTerminalKind,
    WakeTerminalV1,
    WakeV1,
)
from suzka.r19.ledger import WakeState, derive_wake_view

from test_r19_ip13b_contracts import (
    EVENT_A,
    EVENT_C,
    EVENT_R,
    EVENT_T,
    EXPIRES,
    INC,
    NOT_BEFORE,
    TIMING,
    _hold,
    _wake,
)

_K = HistoryOperationKind
S = WakeState
OTHER_EVENT = "00000000-0000-4000-8000-000000000009"

# Claims at 02:00, 02:10, 02:20; each lease is 120 s (TIMING.claim_lease_s).
CLAIM_AT = ("2026-11-16T02:00:00Z", "2026-11-16T02:10:00Z", "2026-11-16T02:20:00Z")
LEASE_END = ("2026-11-16T02:02:00Z", "2026-11-16T02:12:00Z", "2026-11-16T02:22:00Z")


def _rec(
    kind: RecordKind, op: HistoryOperationKind, record: object, sequence: int
) -> AdoptedRecord:
    return AdoptedRecord(op, kind, record, "tx-" + str(sequence), EVENT_A, sequence)


def _wake_rec(wake: WakeV1, sequence: int = 20) -> AdoptedRecord:
    return _rec(RecordKind.WAKE, _K.R19_WAKE_MATERIALIZE, wake, sequence)


def _claim(wake: WakeV1, attempt: int = 1) -> WakeClaimV1:
    return WakeClaimV1.for_wake(
        wake,
        attempt,
        claim_event_id=EVENT_C,
        claim_sequence=20 + attempt,
        claimed_utc=CLAIM_AT[attempt - 1],
        timing=TIMING,
    )


def _claim_rec(claim: WakeClaimV1) -> AdoptedRecord:
    return _rec(RecordKind.WAKE_CLAIM, _K.R19_WAKE_CLAIM, claim, claim.claim_sequence)


def _outcome_rec(
    claim: WakeClaimV1, reason: RefuseReason = RefuseReason.SOURCE_UNAVAILABLE
) -> AdoptedRecord:
    outcome = WakeAttemptOutcomeV1(claim.claim_id, AttemptOutcome.REFUSED, reason)
    return _rec(
        RecordKind.WAKE_ATTEMPT_OUTCOME, _K.R19_WAKE_ATTEMPT_OUTCOME, outcome, 50
    )


def _closure_rec(claim: WakeClaimV1) -> AdoptedRecord:
    receipt = _hold(claim)
    closure = CycleClosureV1(LogicalCycleResolutionV1.of_hold(INC, receipt), receipt)
    return _rec(RecordKind.CYCLE_CLOSURE, _K.R16_CYCLE_RECEIPT, closure, 60)


def _terminal(wake: WakeV1, **changes: object) -> WakeTerminalV1:
    base = WakeTerminalV1.for_wake(
        wake,
        WakeTerminalKind.FAILED_TERMINAL,
        TerminalReason.EXPIRED_WAKE,
        event_id=EVENT_T,
        event_sequence=70,
    )
    return replace(base, **changes)


def _terminal_rec(terminal: WakeTerminalV1) -> AdoptedRecord:
    return _rec(RecordKind.WAKE_TERMINAL, _K.R19_WAKE_TERMINAL, terminal, 70)


def _state(records: list[AdoptedRecord], now: str, wake: WakeV1) -> WakeState:
    entry = derive_wake_view(records, now, clock_floor=None).get(wake.wakeup_id)
    assert entry is not None
    return entry.state


def _three_claims(wake: WakeV1) -> list[AdoptedRecord]:
    return [_wake_rec(wake), *(_claim_rec(_claim(wake, n)) for n in (1, 2, 3))]


# ------------------------------------------------------------------- states


def test_scheduled_due_and_expired_at_their_half_open_boundaries() -> None:
    wake = _wake()
    rows = [_wake_rec(wake)]
    assert _state(rows, "2026-11-16T00:59:59Z", wake) is S.SCHEDULED
    assert _state(rows, NOT_BEFORE, wake) is S.DUE  # not_before <= T
    assert _state(rows, "2026-11-22T23:59:59Z", wake) is S.DUE
    assert _state(rows, EXPIRES, wake) is S.EXPIRED  # T < expires, half-open


def test_claim_lease_is_half_open_and_a_lapsed_claim_is_retryable() -> None:
    wake = _wake()
    first = _claim(wake)
    rows = [_wake_rec(wake), _claim_rec(first)]
    assert first.claim_lease_expires_utc == LEASE_END[0]
    assert _state(rows, "2026-11-16T02:01:59Z", wake) is S.CLAIMED_LIVE
    assert _state(rows, LEASE_END[0], wake) is S.RETRYABLE  # T < lease, half-open
    # An outcome row records a reason; it never shortens the lease (P2-A).
    refused = [*rows, _outcome_rec(first)]
    assert _state(refused, "2026-11-16T02:00:30Z", wake) is S.CLAIMED_LIVE
    assert _state(refused, LEASE_END[0], wake) is S.RETRYABLE
    entry = derive_wake_view(refused, "2026-11-16T02:00:30Z", clock_floor=None).get(
        wake.wakeup_id
    )
    assert entry is not None
    assert (entry.attempts, entry.next_attempt_index, entry.lease_ended) == (
        1,
        2,
        False,
    )
    assert entry.reason == RefuseReason.SOURCE_UNAVAILABLE.value


def test_a_lapsed_claim_of_an_expired_wake_is_expired_never_retryable() -> None:
    wake = _wake()
    rows = [_wake_rec(wake), _claim_rec(_claim(wake))]
    assert _state(rows, EXPIRES, wake) is S.EXPIRED
    assert _state(rows, "2026-11-22T23:59:59Z", wake) is S.RETRYABLE


def test_exhausted_needs_three_claims_and_the_lease_over_or_an_outcome() -> None:
    wake = _wake()
    rows = _three_claims(wake)
    assert (
        _state(rows[:3], "2026-11-16T03:00:00Z", wake) is S.RETRYABLE
    )  # two claims only
    assert _state(rows, "2026-11-16T02:21:59Z", wake) is S.CLAIMED_LIVE
    assert _state(rows, LEASE_END[2], wake) is S.EXHAUSTED
    # The third attempt's outcome row ends the cycle's attempts even inside its lease.
    with_outcome = [*rows, _outcome_rec(_claim(wake, 3))]
    assert _state(with_outcome, "2026-11-16T02:20:30Z", wake) is S.EXHAUSTED
    # EXHAUSTED wins over EXPIRED: ATTEMPTS_EXHAUSTED is the more specific reason.
    assert _state(rows, EXPIRES, wake) is S.EXHAUSTED


def test_expired_pending_lease_is_derived_with_no_row() -> None:
    """#373 A3-3. ``for_wake`` caps a lease at the expiry, so only a decoded /
    hand-built claim can outlive it; the view must still not write or close."""

    wake = _wake()
    late = WakeClaimV1(
        wake.wakeup_id,
        wake.logical_cycle_id,
        wake.evidence_fingerprint,
        wake.incarnation_id,
        1,
        EVENT_C,
        21,
        "2026-11-23T00:01:00Z",
    )
    rows = [_wake_rec(wake), _claim_rec(late)]
    assert _state(rows, "2026-11-23T00:00:30Z", wake) is S.EXPIRED_PENDING_LEASE
    assert _state(rows, "2026-11-23T00:01:00Z", wake) is S.EXPIRED
    # No row was needed: the view is a function of the records and the instant.
    assert len(rows) == 2


def test_closed_and_terminal_states_win_and_a_closed_wake_is_never_expired() -> None:
    wake = _wake()
    first = _claim(wake)
    closed = [_wake_rec(wake), _claim_rec(first), _closure_rec(first)]
    assert _state(closed, "2026-11-16T02:00:30Z", wake) is S.CLOSED
    assert _state(closed, "2027-01-01T00:00:00Z", wake) is S.CLOSED  # past the expiry
    entry = derive_wake_view(closed, "2026-11-16T02:00:30Z", clock_floor=None).get(
        wake.wakeup_id
    )
    assert entry is not None and entry.reason == "HOLD"
    failed = [_wake_rec(wake), _terminal_rec(_terminal(wake))]
    assert _state(failed, "2026-11-16T00:00:00Z", wake) is S.FAILED
    cancelled = _terminal(
        wake,
        terminal=WakeTerminalKind.CANCELLED,
        reason=TerminalReason.SUPERSEDED,
        superseding_wakeup_id=_wake(evidence_fingerprint="e" * 64).wakeup_id,
    )
    assert (
        _state([_wake_rec(wake), _terminal_rec(cancelled)], NOT_BEFORE, wake)
        is S.CANCELLED
    )


def test_entries_are_ordered_for_the_claim_priority() -> None:
    early = _wake(evidence_fingerprint="1" * 64, not_before_utc="2026-11-16T01:00:00Z")
    later = _wake(evidence_fingerprint="2" * 64, not_before_utc="2026-11-16T03:00:00Z")
    same_time_later_cause = _wake(evidence_fingerprint="3" * 64, cause_sequence=99)
    view = derive_wake_view(
        [_wake_rec(later), _wake_rec(same_time_later_cause), _wake_rec(early)],
        "2026-11-16T04:00:00Z",
        clock_floor=None,
    )
    assert [item.wake for item in view.entries] == [early, same_time_later_cause, later]
    assert view.in_state(S.DUE) == view.entries
    assert view.get("wakeup-" + "0" * 64) is None


def test_the_clock_floor_marks_an_unverifiable_instant_without_changing_a_state() -> (
    None
):
    wake = _wake()
    rows = [_wake_rec(wake)]
    behind = derive_wake_view(rows, NOT_BEFORE, clock_floor="2026-11-16T05:00:00Z")
    assert behind.clock_verifiable is False
    assert behind.entries[0].state is S.DUE  # derived from the instant it was given
    at = derive_wake_view(rows, NOT_BEFORE, clock_floor=NOT_BEFORE)
    assert at.clock_verifiable is True
    assert derive_wake_view(rows, NOT_BEFORE, clock_floor=None).clock_verifiable is True
    with pytest.raises(ValueError):
        derive_wake_view(rows, NOT_BEFORE, clock_floor="not a time")
    with pytest.raises(ValueError):
        derive_wake_view(rows, "2026-11-16 00:00:00", clock_floor=None)


# --------------------------------------------------------- consistency (A-4)


def test_a_terminal_with_a_foreign_cycle_id_is_corrupt() -> None:
    """A-4: ``wake_terminals_insert`` requires the Wake's own ``cycle:`` key, so
    a foreign cycle id cannot be written; if one is ever read the Wake is
    flagged and left alone, never closed."""

    wake = _wake()
    foreign = _terminal(wake, logical_cycle_id="9" * 64)
    view = derive_wake_view(
        [_wake_rec(wake), _terminal_rec(foreign)], NOT_BEFORE, clock_floor=None
    )
    entry = view.get(wake.wakeup_id)
    assert entry is not None
    assert (entry.state, entry.reason) == (S.CORRUPT, "TERMINAL_CYCLE_MISMATCH")
    # The writer-side builder cannot produce it.
    assert _terminal(wake).logical_cycle_id == wake.logical_cycle_id


@pytest.mark.parametrize(
    "build, reason",
    [
        (
            lambda w, c: [
                _wake_rec(w),
                _claim_rec(replace(c, logical_cycle_id="9" * 64)),
            ],
            "CLAIM_CYCLE_MISMATCH",
        ),
        (
            lambda w, c: [
                _wake_rec(w),
                _claim_rec(replace(c, evidence_fingerprint="9" * 64)),
            ],
            "CLAIM_CYCLE_MISMATCH",
        ),
        (
            lambda w, c: [_wake_rec(w), _claim_rec(replace(c, attempt_index=2))],
            "CLAIM_INDEX_GAP",
        ),
        (lambda w, c: [_wake_rec(w), _claim_rec(c), _claim_rec(c)], "CLAIM_INDEX_GAP"),
        (lambda w, c: [_wake_rec(w), _wake_rec(w)], "DUPLICATE_WAKE"),
        (
            lambda w, c: [
                _wake_rec(w),
                _claim_rec(c),
                _outcome_rec(c),
                _outcome_rec(c),
            ],
            "DUPLICATE_OUTCOME",
        ),
        (
            lambda w, c: [
                _wake_rec(w),
                _claim_rec(c),
                _closure_rec(c),
                _terminal_rec(_terminal(w)),
            ],
            "TERMINAL_AND_RESOLUTION",
        ),
        (
            lambda w, c: [
                _wake_rec(w),
                _terminal_rec(_terminal(w)),
                _terminal_rec(_terminal(w)),
            ],
            "DUPLICATE_CLOSURE",
        ),
        (lambda w, c: [_wake_rec(w), _closure_rec(c)], "RESOLUTION_NOT_LATEST_CLAIM"),
        (
            lambda w, c: [
                _wake_rec(w),
                _claim_rec(c),
                _outcome_rec(c),
                _closure_rec(c),
            ],
            "RESOLUTION_AND_OUTCOME",
        ),
    ],
)
def test_contradicting_rows_make_the_wake_corrupt(build, reason: str) -> None:  # type: ignore[no-untyped-def]
    wake = _wake()
    claim = _claim(wake)
    view = derive_wake_view(
        build(wake, claim), "2026-11-16T02:00:30Z", clock_floor=None
    )
    entry = view.get(wake.wakeup_id)
    assert entry is not None
    assert (entry.state, entry.reason) == (S.CORRUPT, reason)


def test_a_resolution_must_belong_to_the_latest_claim() -> None:
    wake = _wake()
    first, second = _claim(wake, 1), _claim(wake, 2)
    stale = [
        _wake_rec(wake),
        _claim_rec(first),
        _claim_rec(second),
        _closure_rec(first),
    ]
    entry = derive_wake_view(stale, "2026-11-16T03:00:00Z", clock_floor=None).get(
        wake.wakeup_id
    )
    assert entry is not None and entry.reason == "RESOLUTION_NOT_LATEST_CLAIM"
    latest = [
        _wake_rec(wake),
        _claim_rec(first),
        _claim_rec(second),
        _closure_rec(second),
    ]
    assert _state(latest, "2026-11-16T03:00:00Z", wake) is S.CLOSED


def test_records_that_name_no_wake_are_orphans_not_state() -> None:
    wake = _wake()
    other = _wake(evidence_fingerprint="e" * 64)
    view = derive_wake_view(
        [_wake_rec(wake), _claim_rec(_claim(other)), _terminal_rec(_terminal(other))],
        NOT_BEFORE,
        clock_floor=None,
    )
    assert [item.state for item in view.entries] == [S.DUE]
    assert {kind for kind, _ in view.orphans} == {
        RecordKind.WAKE_CLAIM,
        RecordKind.WAKE_TERMINAL,
    }


def test_a_record_of_the_wrong_type_is_a_programming_error() -> None:
    with pytest.raises(TypeError):
        derive_wake_view(
            [_rec(RecordKind.WAKE, _K.R19_WAKE_MATERIALIZE, object(), 1)],
            NOT_BEFORE,
            clock_floor=None,
        )


def test_other_record_kinds_are_ignored() -> None:
    wake = _wake()
    noise = _rec(RecordKind.ACTION_INTENT, _K.R18_INTENT_ISSUE, object(), 5)
    assert (
        derive_wake_view([noise, _wake_rec(wake)], NOT_BEFORE, clock_floor=None)
        .entries[0]
        .state
        is S.DUE
    )
    assert OTHER_EVENT != EVENT_R  # fixtures stay distinct


# ------------------------------------------------------------------- purity


def test_the_view_is_pure() -> None:
    tree = ast.parse(inspect.getsource(ledger))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imported == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "enum",
        "typing",
        "suzka.r16.cycle_contracts",
        "suzka.r16.history_records",
        "suzka.r16.history_reader",  # TYPE_CHECKING only: the AdoptedRecord annotation
        "suzka.r19.contracts",
    }
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
    source = inspect.getsource(ledger)
    for forbidden in (
        "datetime.now",
        "utcnow",
        "time.time",
        "sqlite3",
        "open(",
        "random",
    ):
        assert forbidden not in source
    # Nothing here writes or reads a store, and no clock is read: the instant is an argument.
    assert "now_utc" in inspect.signature(derive_wake_view).parameters
    assert not [
        name
        for name in dir(ledger)
        if name.startswith("write") or name.startswith("submit")
    ]


# --------------------------------------------- Commitment deadline vs expiry


def _with_deadline(wake: WakeV1, deadline: str) -> WakeV1:
    """``WakeV1`` forbids a deadline before its expiry, so a deadline earlier
    than ``expires_at_utc`` cannot be constructed; the view still takes
    ``min(expires, deadline)`` (defensive, D1 §4-2), exercised by forcing it."""

    object.__setattr__(wake, "commitment_deadline_utc", deadline)
    return wake


def test_the_deadline_equal_to_the_expiry_is_the_end_of_every_window() -> None:
    wake = _wake(commitment_deadline_utc=EXPIRES)
    rows = [_wake_rec(wake)]
    assert _state(rows, "2026-11-22T23:59:59Z", wake) is S.DUE
    assert _state(rows, EXPIRES, wake) is S.EXPIRED
    first = _claim(wake)
    lapsed = [_wake_rec(wake), _claim_rec(first)]
    assert _state(lapsed, "2026-11-22T23:59:59Z", wake) is S.RETRYABLE
    assert _state(lapsed, EXPIRES, wake) is S.EXPIRED
    entry = derive_wake_view(rows, NOT_BEFORE, clock_floor=None).entries[0]
    assert entry.expiry_utc == EXPIRES


def test_a_deadline_after_the_expiry_does_not_extend_it() -> None:
    wake = _wake(commitment_deadline_utc="2026-12-31T00:00:00Z")
    assert _state([_wake_rec(wake)], EXPIRES, wake) is S.EXPIRED


def test_the_view_takes_the_earlier_of_expiry_and_deadline() -> None:
    deadline = "2026-11-17T00:00:00Z"
    wake = _with_deadline(_wake(), deadline)
    rows = [_wake_rec(wake)]
    assert _state(rows, "2026-11-16T23:59:59Z", wake) is S.DUE
    assert _state(rows, deadline, wake) is S.EXPIRED  # DUE -> EXPIRED at the deadline
    lapsed = [_wake_rec(wake), _claim_rec(_claim(wake))]
    assert _state(lapsed, "2026-11-16T23:59:59Z", wake) is S.RETRYABLE
    assert _state(lapsed, deadline, wake) is S.EXPIRED  # RETRYABLE -> EXPIRED
    late = WakeClaimV1(
        wake.wakeup_id,
        wake.logical_cycle_id,
        wake.evidence_fingerprint,
        wake.incarnation_id,
        1,
        EVENT_C,
        21,
        "2026-11-17T00:01:00Z",
    )
    pending = [_wake_rec(wake), _claim_rec(late)]
    assert _state(pending, "2026-11-17T00:00:30Z", wake) is S.EXPIRED_PENDING_LEASE
    assert _state(pending, "2026-11-17T00:01:00Z", wake) is S.EXPIRED


def test_a_wrong_typed_adopted_record_raises_and_says_so() -> None:
    assert "TypeError" in (derive_wake_view.__doc__ or "")
    assert "IP-20d" in (derive_wake_view.__doc__ or "")
