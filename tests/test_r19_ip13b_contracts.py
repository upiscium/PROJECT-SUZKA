"""IP-13b: R19 pure Wake / claim / completion / Outbox / Inbox values (D-R19).

Every value built here is a FIXTURE ONLY: nothing is scheduled, claimed,
persisted or delivered.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from typing import Any

import pytest

import suzka.r19 as r19
from suzka.r16.cycle_contracts import (
    FrozenPredecessor,
    HoldReason,
    HoldReceiptV1,
    LogicalCycleResolutionV1,
    ResultKind,
    wake_review_intent_key,
)
from suzka.r16.history_codec import PolicyKind, PolicyRow
from suzka.r16.successor_policy import (
    Interpretation,
    InterpretationStage,
    LineageState,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
)
from suzka.r18.contracts import GoalOrigin
from suzka.r19 import contracts
from suzka.r19.contracts import (
    INBOX_ASSURANCE,
    MAX_ATTEMPTS,
    MAX_RECORD_BYTES,
    AckStatus,
    AttemptOutcome,
    InboxAckV1,
    InboxReplyV1,
    OutboxMessageV1,
    OutboxReason,
    OutboxTemplate,
    RefuseReason,
    TerminalReason,
    WakeAttemptOutcomeV1,
    WakeCause,
    WakeClaimV1,
    WakeKind,
    WakeTerminalKind,
    WakeTerminalV1,
    WakeTiming,
    WakeV1,
    logical_cycle_id,
    may_supersede,
    next_attempt_index,
    notification_required,
    outbox_reason_for,
    wake_fingerprint,
    wake_window,
    wakeup_id,
)

H = "a" * 64
INC = "b" * 64
FP = "f" * 64
EVENT_A = "00000000-0000-4000-8000-000000000001"
EVENT_C = "00000000-0000-4000-8000-000000000002"
EVENT_R = "00000000-0000-4000-8000-000000000003"
EVENT_T = "00000000-0000-4000-8000-000000000004"
CREATED = "2026-11-16T00:00:00Z"
NOT_BEFORE = "2026-11-16T01:00:00Z"
EXPIRES = "2026-11-23T00:00:00Z"
OUTCOME = PolicyRef("r18-outcome", "outcome-1", 1, H)
COMMITMENT = PolicyRef("commitment", "commitment-1", 1, H)
TIMING = WakeTiming()

CAUSE_REF = {
    WakeCause.GOAL_ADMITTED: PolicyRef("goal-admission", "goal-1", 1, H),
    WakeCause.COMMITMENT_ADMITTED: PolicyRef("commitment-admission", "commitment-1", 1, H),
    WakeCause.FEEDBACK_ADMITTED: PolicyRef("r17-feedback", "feedback-1", 1, H),
    WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK: PolicyRef("r17-assessment", "assessment-1", 1, H),
}
ORIGIN_REFS = {
    WakeCause.GOAL_ADMITTED: (),
    WakeCause.COMMITMENT_ADMITTED: (COMMITMENT,),
    WakeCause.FEEDBACK_ADMITTED: (COMMITMENT, OUTCOME),
    WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK: (COMMITMENT, OUTCOME),
}


def _wake(cause: WakeCause = WakeCause.FEEDBACK_ADMITTED, **changes: Any) -> WakeV1:
    base = WakeV1(
        kind=WakeKind.GOAL_REEVALUATION,
        cause=cause,
        cause_event_id=EVENT_A,
        cause_sequence=20,
        cause_ref=CAUSE_REF[cause],
        goal_id="goal-1",
        goal_revision=1,
        goal_digest=H,
        experience_id="experience-x",
        goal_origin=GoalOrigin.USER_REQUEST,
        origin_refs=ORIGIN_REFS[cause],
        evidence_fingerprint=FP,
        policy_kind=PolicyKind.SUCCESSOR_POLICY_V2,
        frozen_predecessor=FrozenPredecessor.decision("decision-" + "d" * 64, H),
        horizon_sequence=19,
        incarnation_id=INC,
        journal_lineage_id="lineage-1",
        not_before_utc=NOT_BEFORE,
        expires_at_utc=EXPIRES,
        commitment_deadline_utc=None,
        context_id="context-1",
    )
    return replace(base, **changes)


def _claim(wake: WakeV1 | None = None, attempt: int = 1, at: str = "2026-11-16T02:00:00Z") -> WakeClaimV1:
    return WakeClaimV1.for_wake(
        wake or _wake(), attempt, claim_event_id=EVENT_C, claim_sequence=21, claimed_utc=at, timing=TIMING
    )


def _hold(claim: WakeClaimV1) -> HoldReceiptV1:
    return HoldReceiptV1(
        logical_cycle_id=claim.logical_cycle_id,
        review_intent_key=wake_review_intent_key(
            claim.logical_cycle_id, _wake().frozen_predecessor, PolicyKind.SUCCESSOR_POLICY_V2, FP
        ),
        claim_id=claim.claim_id,
        claim_digest=claim.claim_digest,
        claim_event_id=claim.claim_event_id,
        hold_event_id=EVENT_R,
        hold_sequence=22,
        reason=HoldReason.STALE_SOURCE,
        policy_row=None,
        current_fingerprint="9" * 64,
        sealed_input_digest=H,
        prepare_checked_at_utc="2026-11-16T02:00:05Z",
    )


def _tamper(payload: bytes, **changes: object) -> bytes:
    row = json.loads(payload)
    row.update(changes)
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


# --------------------------------------------------------------- identities


def test_wake_and_cycle_ids_use_only_incarnation_goal_kind_and_fingerprint() -> None:
    """#369 A1-1 / A1-3: no event, attempt, claim, clock or cause."""

    for function in (wakeup_id, logical_cycle_id):
        assert tuple(inspect.signature(function).parameters) == (
            "incarnation_id",
            "goal_id",
            "kind",
            "evidence_fingerprint",
        )
    wake = _wake()
    assert wake.wakeup_id == wakeup_id(INC, "goal-1", WakeKind.GOAL_REEVALUATION, FP)
    assert wake.logical_cycle_id == logical_cycle_id(INC, "goal-1", WakeKind.GOAL_REEVALUATION, FP)
    assert wake.wakeup_id[len("wakeup-") :] != wake.logical_cycle_id  # separate domains
    # The cause is provenance: the same evidence from another cause is the same Wake.
    other = _wake(cause=WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK, cause_event_id=EVENT_C, cause_sequence=40)
    assert other.wakeup_id == wake.wakeup_id and other.wake_digest != wake.wake_digest
    for changes in ({"incarnation_id": H}, {"goal_id": "goal-2"}, {"evidence_fingerprint": "e" * 64}):
        assert _wake(**changes).wakeup_id != wake.wakeup_id


def test_wake_fingerprint_needs_a_qualified_input() -> None:
    """D-R19 §3: no Wake from an input with an UNAVAILABLE part."""

    from test_r16_ip8b_successor_policy import _active, _verified

    value = _verified()
    assert wake_fingerprint(value) == value.evidence_fingerprint()
    assert wake_fingerprint(_active()) == _active().evidence_fingerprint()  # ABSENT_INITIAL predecessor
    for changes in (
        {"required_source": RequiredSourceState.UNAVAILABLE},
        {"predecessor": PredecessorState.UNAVAILABLE, "predecessor_ref": None},
        {"lineage": LineageState.UNAVAILABLE},
        {"interpretation": Interpretation.UNAVAILABLE, "interpretation_stage": InterpretationStage.NONE},
    ):
        assert wake_fingerprint(replace(value, **changes)) is None, changes
    # The R15 (stage, interpretation) pair is part of the fingerprint.
    s1 = replace(value, interpretation=Interpretation.SETTLED, interpretation_stage=InterpretationStage.S1)
    s2 = replace(value, interpretation=Interpretation.SETTLED, interpretation_stage=InterpretationStage.S2)
    assert wake_fingerprint(s1) != wake_fingerprint(s2)
    assert wake_fingerprint(replace(value, event_id="event-99", event_sequence=99)) == wake_fingerprint(value)


# --------------------------------------------------------------------- wake


@pytest.mark.parametrize("cause", list(WakeCause))
def test_each_cause_round_trips(cause: WakeCause) -> None:
    wake = _wake(cause)
    payload = wake.canonical_bytes()
    assert WakeV1.from_json(payload) == wake
    assert len(payload) <= MAX_RECORD_BYTES


def test_cause_ref_and_origin_refs_follow_the_cause() -> None:
    with pytest.raises(ValueError, match="cites a r17-feedback"):
        _wake(cause_ref=CAUSE_REF[WakeCause.GOAL_ADMITTED])
    with pytest.raises(ValueError, match="one verified Outcome"):
        _wake(origin_refs=(COMMITMENT,))
    with pytest.raises(ValueError, match="one verified Outcome"):
        _wake(WakeCause.GOAL_ADMITTED, origin_refs=(OUTCOME,))
    with pytest.raises(ValueError, match="one verified Outcome"):
        _wake(origin_refs=(OUTCOME, PolicyRef("r18-outcome", "outcome-2", 1, H)))
    with pytest.raises(ValueError, match="its Commitment"):
        _wake(WakeCause.COMMITMENT_ADMITTED, origin_refs=())
    with pytest.raises(ValueError, match="sorted"):
        _wake(origin_refs=(OUTCOME, COMMITMENT))
    with pytest.raises(ValueError):
        _wake(origin_refs=(COMMITMENT,) * 2 + (OUTCOME,))
    with pytest.raises(ValueError, match="at most one Commitment"):
        _wake(WakeCause.COMMITMENT_ADMITTED, origin_refs=(COMMITMENT, PolicyRef("commitment", "commitment-2", 1, H)))
    assert _wake().verified_outcome_key == ("outcome-1", 1)
    assert _wake(WakeCause.GOAL_ADMITTED).verified_outcome_key is None


def test_wake_window_and_predecessor_horizon() -> None:
    with pytest.raises(ValueError, match="due before"):
        _wake(not_before_utc=EXPIRES)
    with pytest.raises(ValueError, match="Commitment deadline"):
        _wake(commitment_deadline_utc="2026-11-20T00:00:00Z")
    _wake(commitment_deadline_utc=EXPIRES)
    with pytest.raises(ValueError, match="one horizon"):
        _wake(frozen_predecessor=FrozenPredecessor.absence(18))
    _wake(frozen_predecessor=FrozenPredecessor.absence(19))
    assert _wake(goal_origin=GoalOrigin.SUBJECT_MOTIVATION).goal_origin is GoalOrigin.SUBJECT_MOTIVATION


def test_wake_bytes_are_strict() -> None:
    payload = _wake().canonical_bytes()
    for changes in (
        {"wakeup_id": "wakeup-" + "0" * 64},
        {"logical_cycle_id": "0" * 64},
        {"kind": "OTHER"},
        {"schema_version": 2},
    ):
        with pytest.raises((ValueError, TypeError)):
            WakeV1.from_json(_tamper(payload, **changes))
    with pytest.raises(ValueError):
        WakeV1.from_json(b" " + payload)


# ----------------------------------------------------------------- schedule


def test_wake_window_follows_the_hp5_numbers() -> None:
    def window(cause: WakeCause, **kw: Any) -> tuple[str, str] | None:
        args: dict[str, Any] = {
            "cause_utc": CREATED,
            "created_utc": CREATED,
            "goal_deadline_utc": None,
            "commitment_deadline_utc": None,
            "timing": TIMING,
        }
        args.update(kw)
        return wake_window(cause, **args)

    assert window(WakeCause.GOAL_ADMITTED) == ("2026-11-16T01:00:00Z", "2026-11-23T00:00:00Z")
    assert window(WakeCause.FEEDBACK_ADMITTED) == ("2026-11-16T00:01:00Z", "2026-11-23T00:00:00Z")
    # A Commitment due in 30 min: not_before is pulled back to deadline − 10 min.
    assert window(WakeCause.COMMITMENT_ADMITTED, commitment_deadline_utc="2026-11-16T00:30:00Z") == (
        "2026-11-16T00:20:00Z",
        "2026-11-16T00:30:00Z",
    )
    # Expiry is the earliest of 7 days and the deadlines.
    assert window(WakeCause.GOAL_ADMITTED, goal_deadline_utc="2026-11-18T00:00:00Z") == (
        "2026-11-16T01:00:00Z",
        "2026-11-18T00:00:00Z",
    )
    # GOAL due after its deadline − margin: not schedulable, never silently late.
    assert window(WakeCause.GOAL_ADMITTED, goal_deadline_utc="2026-11-16T01:05:00Z") is None
    # FEEDBACK with under a margin left: not schedulable.
    assert window(WakeCause.FEEDBACK_ADMITTED, commitment_deadline_utc="2026-11-16T00:05:00Z") is None
    # A late materialize keeps the cause-based not_before (already due).
    assert window(WakeCause.GOAL_ADMITTED, created_utc="2026-11-16T05:00:00Z") == (
        "2026-11-16T01:00:00Z",
        "2026-11-23T05:00:00Z",
    )
    # GOAL / COMMITMENT keep the literal rule: a late materialize near the
    # deadline is still made, already due.
    assert window(
        WakeCause.GOAL_ADMITTED, created_utc="2026-11-16T02:00:00Z", goal_deadline_utc="2026-11-16T02:05:00Z"
    ) == ("2026-11-16T01:00:00Z", "2026-11-16T02:05:00Z")
    # The extra rule applies to the two Outcome causes only.
    assert window(WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK, commitment_deadline_utc="2026-11-16T00:05:00Z") is None
    with pytest.raises(ValueError, match="at or after"):
        window(WakeCause.GOAL_ADMITTED, cause_utc="2026-11-16T05:00:00Z")


def test_timing_refuses_zero_lease_or_tick() -> None:
    assert (TIMING.claim_lease_s, TIMING.tick_s, TIMING.deadline_margin_s) == (120, 30, 600)
    for changes in ({"claim_lease_s": 0}, {"tick_s": 0}, {"goal_wake_delay_s": -1}):
        with pytest.raises(ValueError):
            WakeTiming(**changes)


# -------------------------------------------------------------------- claim


def test_claim_binds_the_wake_and_bounds_its_lease() -> None:
    wake = _wake()
    claim = _claim(wake)
    assert (claim.wakeup_id, claim.logical_cycle_id, claim.evidence_fingerprint, claim.incarnation_id) == (
        wake.wakeup_id,
        wake.logical_cycle_id,
        FP,
        INC,
    )
    assert claim.claim_lease_expires_utc == "2026-11-16T02:02:00Z"
    assert claim.claim_id.startswith("wake-claim-")
    assert WakeClaimV1.from_json(claim.canonical_bytes()) == claim
    # Attempts of one cycle share the cycle but not the claim.
    second = _claim(wake, 2, "2026-11-16T03:00:00Z")
    assert second.logical_cycle_id == claim.logical_cycle_id and second.claim_id != claim.claim_id
    # The lease never outlives the Wake or the Commitment deadline.
    short = _wake(commitment_deadline_utc="2026-11-16T02:01:00Z", expires_at_utc="2026-11-16T02:01:00Z")
    assert _claim(short).claim_lease_expires_utc == "2026-11-16T02:01:00Z"


def test_claim_refuses_outside_the_window_and_bad_attempts() -> None:
    for at in ("2026-11-16T00:59:59Z", EXPIRES):
        with pytest.raises(ValueError, match="not due"):
            _claim(at=at)
    for attempt in (0, MAX_ATTEMPTS + 1):
        with pytest.raises(ValueError):
            _claim(attempt=attempt)
    assert [next_attempt_index(n) for n in range(4)] == [1, 2, 3, None]


def test_in_window_checks_all_three_deadlines() -> None:
    """#373 A4-4: WakeExpiry, ClaimLease, U2cDeadline."""

    wake = _wake(commitment_deadline_utc=EXPIRES)
    claim = _claim(wake)
    assert claim.in_window(wake, "2026-11-16T02:01:59Z")
    assert not claim.in_window(wake, "2026-11-16T02:02:00Z")  # lease
    assert not claim.in_window(wake, "2026-11-16T00:30:00Z")  # before not_before
    with pytest.raises(ValueError, match="another Wake"):
        claim.in_window(_wake(evidence_fingerprint="e" * 64), "2026-11-16T02:01:00Z")


# ---------------------------------------------------------- attempt / terminal


def test_attempt_outcome_reason_exactly_for_refused() -> None:
    claim = _claim()
    for reason in RefuseReason:
        value = WakeAttemptOutcomeV1(claim.claim_id, AttemptOutcome.REFUSED, reason)
        assert WakeAttemptOutcomeV1.from_json(value.canonical_bytes()) == value
    for outcome in (AttemptOutcome.DEFERRED_BUDGET, AttemptOutcome.LEASE_EXPIRED):
        WakeAttemptOutcomeV1(claim.claim_id, outcome, None)
        with pytest.raises(ValueError):
            WakeAttemptOutcomeV1(claim.claim_id, outcome, RefuseReason.CLOCK_UNVERIFIABLE)
    with pytest.raises(ValueError):
        WakeAttemptOutcomeV1(claim.claim_id, AttemptOutcome.REFUSED, None)


def test_terminal_kinds_and_reasons_are_paired() -> None:
    wake = _wake()
    newer = _wake(evidence_fingerprint="e" * 64)
    for reason in (TerminalReason.EXPIRED_WAKE, TerminalReason.ATTEMPTS_EXHAUSTED):
        value = WakeTerminalV1(wake.wakeup_id, wake.logical_cycle_id, WakeTerminalKind.FAILED_TERMINAL, reason, None, EVENT_T, 30)
        assert WakeTerminalV1.from_json(value.canonical_bytes()) == value
    cancelled = WakeTerminalV1(
        wake.wakeup_id, wake.logical_cycle_id, WakeTerminalKind.CANCELLED, TerminalReason.SUPERSEDED, newer.wakeup_id, EVENT_T, 30
    )
    assert WakeTerminalV1.from_json(cancelled.canonical_bytes()) == cancelled
    assert cancelled.terminal_id == replace(cancelled, event_sequence=31).terminal_id  # one per Wake
    for bad in (
        (WakeTerminalKind.FAILED_TERMINAL, TerminalReason.SUPERSEDED, newer.wakeup_id),
        (WakeTerminalKind.CANCELLED, TerminalReason.EXPIRED_WAKE, None),
        (WakeTerminalKind.CANCELLED, TerminalReason.SUPERSEDED, None),
        (WakeTerminalKind.CANCELLED, TerminalReason.SUPERSEDED, wake.wakeup_id),
        (WakeTerminalKind.FAILED_TERMINAL, TerminalReason.EXPIRED_WAKE, newer.wakeup_id),
    ):
        with pytest.raises(ValueError):
            WakeTerminalV1(wake.wakeup_id, wake.logical_cycle_id, *bad, EVENT_T, 30)


# ------------------------------------------------------------ supersession


def test_supersession_needs_all_four_conditions() -> None:
    """#369 A2-3: same Goal revision, same U2c, not later than the old
    expiry, same Context; and only a FEEDBACK Wake supersedes."""

    old = _wake(WakeCause.COMMITMENT_ADMITTED, commitment_deadline_utc=EXPIRES)
    new = _wake(evidence_fingerprint="e" * 64, commitment_deadline_utc=EXPIRES)
    assert may_supersede(old, new)
    assert may_supersede(_wake(WakeCause.GOAL_ADMITTED), new)  # the old Wake had no U2c
    assert not may_supersede(old, old)
    for changes in (
        {"goal_revision": 2},
        {"commitment_deadline_utc": "2026-11-22T00:00:00Z", "expires_at_utc": "2026-11-22T00:00:00Z"},
        {"origin_refs": (PolicyRef("commitment", "commitment-2", 1, H), OUTCOME)},
        {"context_id": "context-2"},
        {"incarnation_id": H},
    ):
        assert not may_supersede(old, replace(new, **changes)), changes
    later_old = replace(old, expires_at_utc="2026-11-16T02:00:00Z")
    assert not may_supersede(later_old, replace(new, not_before_utc="2026-11-16T03:00:00Z"))
    outcome_wake = _wake(WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK, evidence_fingerprint="e" * 64)
    assert not may_supersede(old, outcome_wake)


def test_a_commitment_wake_supersedes_only_an_older_goal_wake() -> None:
    """IP-20 D1 A-11 / HD-20-5 (b): the Goal-then-Commitment case. A new
    COMMITMENT_ADMITTED Wake cancels an unclaimed GOAL_ADMITTED Wake of the same
    Goal under the same four conditions; no other cause pair changes."""

    goal = _wake(WakeCause.GOAL_ADMITTED)
    commitment = _wake(
        WakeCause.COMMITMENT_ADMITTED, evidence_fingerprint="e" * 64, commitment_deadline_utc=EXPIRES
    )
    assert may_supersede(goal, commitment)
    # The four conditions still hold for the new pair.
    for changes in (
        {"goal_revision": 2},
        {"context_id": "context-2"},
        {"incarnation_id": H},
        {
            "not_before_utc": "2026-11-30T00:00:00Z",
            "expires_at_utc": "2026-12-01T00:00:00Z",
            "commitment_deadline_utc": "2026-12-01T00:00:00Z",
        },
    ):
        assert not may_supersede(goal, replace(commitment, **changes)), changes
    assert not may_supersede(goal, goal)
    # Not the other way round, and not over another Commitment / Outcome Wake.
    assert not may_supersede(commitment, goal)
    assert not may_supersede(
        _wake(WakeCause.COMMITMENT_ADMITTED, commitment_deadline_utc=EXPIRES), commitment
    )
    assert not may_supersede(_wake(WakeCause.FEEDBACK_ADMITTED), commitment)
    assert not may_supersede(
        _wake(WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK, evidence_fingerprint="d" * 64), commitment
    )
    # A new GOAL_ADMITTED Wake never supersedes anything.
    assert not may_supersede(commitment, replace(goal, evidence_fingerprint="d" * 64))


# --------------------------------------------------------------- completion


def test_every_resolution_requires_one_notification() -> None:
    """IP-14b D1 r2 D-2: WakeCompletionV1 is gone; whether a resolved cycle
    is notified is a pure function of its resolution (each result kind has
    its own template)."""

    wake = _wake()
    claim = _claim(wake)
    held = LogicalCycleResolutionV1.of_hold(INC, _hold(claim))
    for kind, result_id in (
        (ResultKind.HOLD, held.result_id),
        (ResultKind.EXISTING_DECISION, "cycle-existing-" + "e" * 64),
        (ResultKind.DECISION_ISSUED, "decision-" + "d" * 64),
    ):
        assert notification_required(replace(held, result_kind=kind, result_id=result_id))
    with pytest.raises(TypeError):
        notification_required(_hold(claim))  # type: ignore[arg-type]
    assert not hasattr(contracts, "WakeCompletionV1") and not hasattr(contracts, "COMPLETION_ID_PREFIX")


def test_an_unknown_result_kind_has_no_notification_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review P3: a closed set; a kind without a template is an error, not False."""

    resolution = LogicalCycleResolutionV1.of_hold(INC, _hold(_claim()))
    monkeypatch.setattr(contracts, "_RESULT_TEMPLATE", {})
    with pytest.raises(ValueError, match="no notification rule"):
        notification_required(resolution)


def test_a_terminal_built_for_a_wake_carries_its_cycle() -> None:
    wake = _wake()
    newer = _wake(evidence_fingerprint="e" * 64)
    failed = WakeTerminalV1.for_wake(wake, WakeTerminalKind.FAILED_TERMINAL, TerminalReason.EXPIRED_WAKE, event_id=EVENT_T, event_sequence=30)
    assert (failed.wakeup_id, failed.logical_cycle_id) == (wake.wakeup_id, wake.logical_cycle_id)
    cancelled = WakeTerminalV1.for_wake(wake, WakeTerminalKind.CANCELLED, TerminalReason.SUPERSEDED, event_id=EVENT_T, event_sequence=30, superseding=newer)
    assert cancelled.superseding_wakeup_id == newer.wakeup_id
    with pytest.raises(ValueError, match="incarnation"):
        WakeTerminalV1.for_wake(wake, WakeTerminalKind.CANCELLED, TerminalReason.SUPERSEDED, event_id=EVENT_T, event_sequence=30, superseding=_wake(evidence_fingerprint="e" * 64, incarnation_id=H))
    with pytest.raises(ValueError):
        WakeTerminalV1.for_wake(wake, WakeTerminalKind.CANCELLED, TerminalReason.EXPIRED_WAKE, event_id=EVENT_T, event_sequence=30)
    with pytest.raises(TypeError):
        WakeTerminalV1.for_wake(wake.wakeup_id, WakeTerminalKind.FAILED_TERMINAL, TerminalReason.EXPIRED_WAKE, event_id=EVENT_T, event_sequence=30)  # type: ignore[arg-type]


def test_a_terminal_names_its_wakes_cycle() -> None:
    wake = _wake()
    failed = WakeTerminalV1(wake.wakeup_id, wake.logical_cycle_id, WakeTerminalKind.FAILED_TERMINAL, TerminalReason.EXPIRED_WAKE, None, EVENT_T, 30)
    assert json.loads(failed.canonical_bytes())["logical_cycle_id"] == wake.logical_cycle_id
    with pytest.raises(ValueError):
        replace(failed, logical_cycle_id="x")
    other_cycle = replace(failed, logical_cycle_id="c" * 64)
    with pytest.raises(ValueError, match="cycle"):
        OutboxMessageV1.for_failed_cycle(wake, other_cycle, created_utc=CREATED, expires_utc=EXPIRES)


def test_the_inbox_assurance_is_the_api_client_claim() -> None:
    """IP-14b D1 r2 §4-2 (P2-2): the SO_PEERCRED socket was dropped, so an
    ACK claims only that a client the SUZKA API admits acknowledged."""

    assert INBOX_ASSURANCE == "SUZKA_API_CLIENT_V1"
    source = inspect.getsource(contracts)
    assert "neither proof that a\n# person read it nor an approval" in source
    assert "LOCAL_OPERATOR_PEER_UID_V1" not in source


# ------------------------------------------------------------------- outbox


def test_outbox_reason_codes_are_coarse() -> None:
    assert outbox_reason_for(ResultKind.HOLD, None) is OutboxReason.HELD
    assert outbox_reason_for(ResultKind.EXISTING_DECISION, None) is OutboxReason.CONFIRMED
    for row in PolicyRow:
        expected = OutboxReason.NO_ACTION if row in (PolicyRow.ROW_5, PolicyRow.ROW_5B) else OutboxReason.PLAN_SELECTED
        assert outbox_reason_for(ResultKind.DECISION_ISSUED, row) is expected
    with pytest.raises(ValueError):
        outbox_reason_for(ResultKind.DECISION_ISSUED, None)
    with pytest.raises(ValueError):
        outbox_reason_for(ResultKind.HOLD, PolicyRow.ROW_3)


def _held(wake: WakeV1 | None = None) -> LogicalCycleResolutionV1:
    return LogicalCycleResolutionV1.of_hold(INC, _hold(_claim(wake)))


def _message(wake: WakeV1 | None = None, **kw: Any) -> OutboxMessageV1:
    wake = wake or _wake()
    args: dict[str, Any] = {"created_utc": CREATED, "expires_utc": EXPIRES}
    args.update(kw)
    return OutboxMessageV1.for_completion(wake, _held(wake), None, **args)


def test_outbox_message_carries_no_row_or_hold_reason() -> None:
    message = _message()
    payload = message.canonical_bytes()
    assert OutboxMessageV1.from_json(payload) == message
    assert (message.template, message.reason_code) == (OutboxTemplate.DECISION_HELD, OutboxReason.HELD)
    assert (message.context_id, message.incarnation_id) == ("context-1", INC)  # from the Wake
    assert b"STALE_SOURCE" not in payload and b"policy_row" not in payload
    assert message.message_id.startswith("outbox-")


def test_outbox_message_belongs_to_the_wakes_cycle() -> None:
    other = _wake(evidence_fingerprint="e" * 64)
    with pytest.raises(ValueError, match="this Wake's cycle"):
        OutboxMessageV1.for_completion(other, _held(), None, created_utc=CREATED, expires_utc=EXPIRES)
    failed = WakeTerminalV1(_wake().wakeup_id, _wake().logical_cycle_id, WakeTerminalKind.FAILED_TERMINAL, TerminalReason.EXPIRED_WAKE, None, EVENT_T, 30)
    with pytest.raises(ValueError, match="this Wake's"):
        OutboxMessageV1.for_failed_cycle(other, failed, created_utc=CREATED, expires_utc=EXPIRES)


def test_outbox_dedupe_is_stable_and_separates_results() -> None:
    wake = _wake()
    claim = _claim(wake)
    first = _message(wake)
    # A crash-and-rescan rebuild later has the same dedupe key and id.
    again = _message(wake, created_utc="2026-11-16T05:00:00Z", expires_utc="2026-11-23T05:00:00Z")
    assert (again.dedupe_key, again.message_id) == (first.dedupe_key, first.message_id)
    assert _message(_wake(context_id="context-2")).dedupe_key != first.dedupe_key
    key = _held(wake).review_intent_key
    issued = LogicalCycleResolutionV1(
        INC, wake.logical_cycle_id, key, claim.claim_id, claim.claim_digest,
        ResultKind.DECISION_ISSUED, "decision-" + "d" * 64, "1" * 64,
    )
    confirmed = LogicalCycleResolutionV1(
        INC, wake.logical_cycle_id, key, claim.claim_id, claim.claim_digest,
        ResultKind.EXISTING_DECISION, "cycle-existing-" + "e" * 64, "3" * 64,
    )

    def make(res: LogicalCycleResolutionV1, row: PolicyRow | None = None) -> OutboxMessageV1:
        return OutboxMessageV1.for_completion(wake, res, row, created_utc=CREATED, expires_utc=EXPIRES)

    # An EXISTING confirmation is never absorbed by the original DECISION_ISSUED.
    assert make(issued, PolicyRow.ROW_4).dedupe_key != make(confirmed).dedupe_key
    assert make(issued, PolicyRow.ROW_5).reason_code is OutboxReason.NO_ACTION


def test_outbox_message_validates_template_and_ttl() -> None:
    message = _message()
    with pytest.raises(ValueError, match="reason code"):
        replace(message, reason_code=OutboxReason.PLAN_SELECTED)
    for expires in (CREATED, "2026-11-23T00:00:01Z"):
        with pytest.raises(ValueError, match="7 days"):
            replace(message, expires_utc=expires)
    wake = _wake()
    failed = WakeTerminalV1(wake.wakeup_id, wake.logical_cycle_id, WakeTerminalKind.FAILED_TERMINAL, TerminalReason.ATTEMPTS_EXHAUSTED, None, EVENT_T, 30)
    cycle_failed = OutboxMessageV1.for_failed_cycle(wake, failed, created_utc=CREATED, expires_utc=EXPIRES)
    assert (cycle_failed.source_id, cycle_failed.reason_code) == (failed.terminal_id, OutboxReason.FAILED)
    assert OutboxMessageV1.from_json(cycle_failed.canonical_bytes()) == cycle_failed
    cancelled = replace(failed, terminal=WakeTerminalKind.CANCELLED, reason=TerminalReason.SUPERSEDED, superseding_wakeup_id=_wake(evidence_fingerprint="e" * 64).wakeup_id)
    with pytest.raises(ValueError, match="FAILED_TERMINAL"):
        OutboxMessageV1.for_failed_cycle(wake, cancelled, created_utc=CREATED, expires_utc=EXPIRES)
    with pytest.raises(ValueError):
        replace(message, template=OutboxTemplate.CYCLE_FAILED, reason_code=OutboxReason.FAILED)


# -------------------------------------------------------------------- inbox


def _message_id() -> str:
    return _message().message_id


def test_ack_is_one_state_transition_per_message() -> None:
    ack = InboxAckV1(_message_id(), "context-1", "session-1", AckStatus.ACKNOWLEDGED, EVENT_T, 40, CREATED)
    payload = ack.canonical_bytes()
    assert InboxAckV1.from_json(payload) == ack
    assert json.loads(payload)["inbox_assurance"] == INBOX_ASSURANCE
    replay = replace(ack, event_id=EVENT_R, event_sequence=41)
    assert replay.ack_id == ack.ack_id and not ack.conflicts_with(replay)
    assert ack.conflicts_with(replace(ack, status=AckStatus.REJECTED))
    assert ack.conflicts_with(replace(ack, context_id="context-2"))
    with pytest.raises(ValueError, match="assurance"):
        InboxAckV1.from_json(_tamper(payload, inbox_assurance="LAN_PEER"))
    with pytest.raises(ValueError):
        InboxAckV1("message-1", "context-1", "session-1", AckStatus.ACKNOWLEDGED, EVENT_T, 40, CREATED)


def test_reply_is_an_observation_reference() -> None:
    reply = InboxReplyV1(_message_id(), "context-1", "session-1", EVENT_T, 40, PolicyRef("observation", "obs-1", 1, H))
    assert InboxReplyV1.from_json(reply.canonical_bytes()) == reply
    assert reply.reply_id == replace(reply, received_sequence=41).reply_id  # same event replayed
    assert reply.reply_id != replace(reply, received_event_id=EVENT_R).reply_id
    with pytest.raises(ValueError, match="observation"):
        replace(reply, observation_ref=PolicyRef("goal", "goal-1", 1, H))


# ------------------------------------------------------- domain separation


def test_every_id_and_digest_is_distinct() -> None:
    """D-R19 §7: each type has its own id and digest domain."""

    wake = _wake()
    claim = _claim(wake)
    hold = _hold(claim)
    resolution = LogicalCycleResolutionV1.of_hold(INC, hold)
    terminal = WakeTerminalV1(wake.wakeup_id, wake.logical_cycle_id, WakeTerminalKind.FAILED_TERMINAL, TerminalReason.EXPIRED_WAKE, None, EVENT_T, 30)
    attempt = WakeAttemptOutcomeV1(claim.claim_id, AttemptOutcome.REFUSED, RefuseReason.CLOCK_UNVERIFIABLE)
    message = _message(wake)
    ack = InboxAckV1(message.message_id, "context-1", "session-1", AckStatus.ACKNOWLEDGED, EVENT_T, 40, CREATED)
    reply = InboxReplyV1(message.message_id, "context-1", "session-1", EVENT_T, 40, PolicyRef("observation", "obs-1", 1, H))
    hexes = [
        wake.wakeup_id, wake.logical_cycle_id, wake.wake_digest,
        claim.claim_id, claim.claim_digest, attempt.attempt_digest,
        hold.hold_id, hold.hold_digest, hold.review_intent_key,
        resolution.resolution_id, resolution.resolution_digest,
        terminal.terminal_id, terminal.terminal_digest,
        message.message_id, message.message_digest, message.dedupe_key,
        ack.ack_id, ack.ack_digest, reply.reply_id, reply.reply_digest,
    ]
    tails = [value[-64:] for value in hexes]
    assert len(set(tails)) == len(tails)


@pytest.mark.parametrize(
    "decode",
    [
        WakeV1.from_json, WakeClaimV1.from_json, WakeAttemptOutcomeV1.from_json, WakeTerminalV1.from_json,
        OutboxMessageV1.from_json, InboxAckV1.from_json, InboxReplyV1.from_json,
    ],
)
def test_every_decoder_refuses_oversize_and_non_bytes(decode: Any) -> None:
    with pytest.raises(ValueError, match="byte bound"):
        decode(b"{" + b" " * MAX_RECORD_BYTES + b"}")
    with pytest.raises(TypeError):
        decode("{}")


# ------------------------------------------------------------------ purity


def test_r19_imports_stay_pure() -> None:
    tree = ast.parse(inspect.getsource(contracts))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {
        "__future__",
        "dataclasses",
        "datetime",
        "enum",
        "typing",
        "suzka.limits",
        "suzka.r16.contracts",
        "suzka.r16.cycle_contracts",
        "suzka.r16.history_codec",
        "suzka.r16.successor_policy",
        "suzka.r17.contracts",
        "suzka.r18.contracts",
    }
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
    # ``ledger`` (IP-20a) and ``materializer`` (IP-20b) are submodules, imported by module, not re-exported.
    assert set(r19.__all__) == {
        name
        for name in dir(r19)
        if not name.startswith("_")
        and name not in {"contracts", "ledger", "materializer", "claim_owner", "review_owner", "raw_state", "scheduler"}
    }
