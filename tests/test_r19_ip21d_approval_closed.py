"""IP-21d: the APPROVAL_CLOSED Wake cause (IP-21 D1 r3 §7, §10, §12-1; HD-21-5 (a)).

The cause chains (Goal, Commitment, real row-4 Decision, real Intent and
forecast) are the IP-16b ones; the verdict / challenge / expiry rows are
fixtures (the real executor is IP-18's).  Every Wake is written by the real
``WakeMaterializerV1`` through the real R07 runtime, read back through
``adopted_records()``.  ``fx.Time`` moves the runtime clock (``requested_at``,
the frozen T) forward for the window closure.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

import r19_wake_fixtures as fx
from suzka.r16.decision_issuer import DecisionIssueStatus
from suzka.r16.history_codec import PolicyKind, PolicyRow
from suzka.r16.history_reader import HeadBlockReason, SubjectHistoryReader
from suzka.r16.history_records import HistoryOperationKind as K
from suzka.r16.successor_policy import (
    LedgerState,
    PolicyRef,
    SuccessorRow,
    evaluate,
)
from suzka.r18.contracts import PolicyVerdict
from suzka.r18.ledger import INTENT_REF_KIND as R18_INTENT_REF_KIND
from suzka.r19 import contracts as contracts_module
from suzka.r19 import materializer as materializer_module
from suzka.r19.contracts import (
    COMMITMENT_REF_KIND,
    INTENT_REF_KIND,
    TerminalReason,
    WakeCause,
    WakeTiming,
    WakeV1,
    may_supersede,
    wake_window,
)
from suzka.r19.materializer import MaterializeStatus, SkipReason
from suzka.r19.review_owner import ReviewStatus
from test_r16_ip10_decision_issuer import _issue
import test_r16_ip16b_ledger_reader as ip16b
import test_r16_ip8b_successor_policy as ip8b
import test_r18_ip16a2_intent_issuer as ip16a2
import test_r19_ip13b_contracts as ip13b
from test_r16_u2a_source_owner import _app

UTC = timezone.utc


# ------------------------------------------------------------------ helpers


def _settings(root, *, r19: bool = True):  # type: ignore[no-untyped-def]
    """R18 on (the IP-16a2 settings) with R19 on or off."""

    base = ip16a2._settings(root)
    return base.model_copy(update={"r19": base.r19.model_copy(update={"enabled": r19})})


@pytest.fixture
def world(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    app = _app(_settings(tmp_path / "w"))
    app.state.r19_scheduler_enabled = False  # the tests scan by hand
    with TestClient(app) as client:
        state = client.app.state
        clock = fx.Time(state, monkeypatch)
        gid, _decision, intent, forecast = ip16b._forward(state)
        yield SimpleNamespace(
            state=state, time=clock, gid=gid, intent=intent, forecast=forecast, monkeypatch=monkeypatch
        )


def _scan(state):  # type: ignore[no-untyped-def]
    return state.wake_materializer.submit_scan().result(timeout=30).value


def _skips(result) -> list[tuple[WakeCause, SkipReason]]:  # type: ignore[no-untyped-def]
    return list(result.skipped)


def _wakes(state) -> list[WakeV1]:  # type: ignore[no-untyped-def]
    return fx.adopted(state, WakeV1)


def _only_approval(state) -> None:  # type: ignore[no-untyped-def]
    """Hide the Goal / Commitment admissions (they are still un-Woken and would
    be tried first, oldest cause first; D1 §12-1 has its own test below)."""

    state.wake_materializer._admission_candidates = lambda: []


def _deny(world) -> None:  # type: ignore[no-untyped-def]
    ip16b._verdict(world.state, world.intent, PolicyVerdict.DENY)


def _three_expiries(world) -> None:  # type: ignore[no-untyped-def]
    verdict = ip16b._verdict(world.state, world.intent)
    for attempt in (1, 2, 3):
        challenge = ip16b._challenge(world.state, world.intent, verdict, attempt, world.forecast)
        ip16b._expire(world.state, challenge)


def _past_window(world) -> None:  # type: ignore[no-untyped-def]
    left = (world.intent.approval_window_end - world.time.now()).total_seconds()
    world.time.advance(max(left, 0) + 5)


def _closure_wake(world, close) -> WakeV1:  # type: ignore[no-untyped-def]
    state = world.state
    _only_approval(state)
    close(world)
    result = _scan(state)
    assert result.status is MaterializeStatus.COMMITTED, _skips(result)
    return result.wake


# ----------------------------------------- the three closures make one Wake


@pytest.mark.parametrize("close", (_deny, _three_expiries, _past_window), ids=("deny", "expiries", "window"))
def test_each_closure_makes_exactly_one_wake_with_the_intent_as_cause(world, close) -> None:  # type: ignore[no-untyped-def]
    state, intent = world.state, world.intent
    _only_approval(state)
    # Open: no Wake.
    quiet = _scan(state)
    assert (quiet.status, _skips(quiet)) == (MaterializeStatus.NONE_LEFT, [])
    close(world)
    result = _scan(state)
    assert result.status is MaterializeStatus.COMMITTED
    wake = result.wake
    assert wake.cause is WakeCause.APPROVAL_CLOSED
    assert wake.goal_id == world.gid
    # The cause event is the Intent's admission; the ref is the R18 Intent ref.
    assert (wake.cause_event_id, wake.cause_sequence) == (
        intent.admitting_event_id,
        intent.admitting_sequence,
    )
    assert wake.cause_ref == PolicyRef(R18_INTENT_REF_KIND, intent.intent_id, 1, intent.intent_digest)
    # No Outcome ref; the Goal's Commitment ref and deadline are cited.
    assert [ref.kind for ref in wake.origin_refs] == [COMMITMENT_REF_KIND]
    assert wake.verified_outcome_key is None
    assert wake.commitment_deadline_utc is not None
    assert wake.not_before_utc < wake.expires_at_utc <= wake.commitment_deadline_utc
    assert fx.adopted(state, WakeV1) == [wake]
    # One Wake per Intent: the same cause event is already woken.
    again = _scan(state)
    assert (again.status, _skips(again)) == (
        MaterializeStatus.NONE_LEFT,
        [(WakeCause.APPROVAL_CLOSED, SkipReason.CAUSE_ALREADY_WOKEN)],
    )
    assert len(_wakes(state)) == 1
    ip16a2._no_failed_events(state)


def test_the_wake_is_timed_like_feedback_from_its_creation(world) -> None:  # type: ignore[no-untyped-def]
    """FEEDBACK / OUTCOME timing: creation + ``feedback_wake_delay_s`` (60 s), not
    a delay from the Intent's admission."""

    _past_window(world)  # the Intent was admitted a (fake) hour ago
    wake = _closure_wake(world, lambda _world: None)
    due = datetime.strptime(wake.not_before_utc, fx.STAMP).replace(tzinfo=UTC)
    assert abs((due - (world.time.now() + timedelta(seconds=60))).total_seconds()) < 10


def test_an_open_or_claimed_intent_makes_no_wake(world) -> None:  # type: ignore[no-untyped-def]
    """(No inverse case: inverse Intents are excluded by type and not tested.)
    INTENT_ISSUED (open) and CLAIMED (a claimed Intent is an Outcome-cause
    Intent) are not closed unclaimed states."""

    state = world.state
    _only_approval(state)
    verdict = ip16b._verdict(state, world.intent)
    assert _scan(state).status is MaterializeStatus.NONE_LEFT  # ALLOW only: still open
    challenge = ip16b._challenge(state, world.intent, verdict, 1, world.forecast)
    ip16b._claim(state, challenge)
    result = _scan(state)
    assert (result.status, _skips(result)) == (MaterializeStatus.NONE_LEFT, [])
    # Even past the window a claimed Intent is not "closed unclaimed".
    _past_window(world)
    result = _scan(state)
    assert (result.status, _skips(result)) == (MaterializeStatus.NONE_LEFT, [])
    assert _wakes(state) == []


def test_a_successor_decision_before_the_scan_makes_no_wake(world) -> None:  # type: ignore[no-untyped-def]
    """The Intent's Decision is no longer the head: the closure was already
    answered (D1 §7-3 "Still the head")."""

    state = world.state
    _only_approval(state)
    _deny(world)
    issued = _issue(state, world.gid)
    assert (issued.status, issued.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_2C)
    result = _scan(state)
    assert result.status is MaterializeStatus.NONE_LEFT
    assert _skips(result) == [(WakeCause.APPROVAL_CLOSED, SkipReason.SUPERSEDED_DECISION)]
    assert _wakes(state) == []


def test_the_wake_reviews_to_row_2c_with_its_closure(world) -> None:  # type: ignore[no-untyped-def]
    """V3: a closed approval evaluates to row 2c (Plan A, no WRITE), and the
    Outbox reason for every row but 5 / 5b is PLAN_SELECTED, so the Wake is
    reported as "made a new decision" (D1 §7-3)."""

    from suzka.r19.contracts import outbox_reason_for
    from suzka.r19.ledger import WakeState
    from suzka.r19.contracts import OutboxReason, ResultKind

    state, clock = world.state, world.time
    wake = _closure_wake(world, _deny)
    assert wake.policy_kind is PolicyKind.SUCCESSOR_POLICY_V3
    clock.advance(120)  # past not_before
    claimed = fx.claim(state, wake)
    result = fx.review(state, claimed.claim.claim_id)
    assert result.status is ReviewStatus.DECISION_ISSUED
    entry = fx.entry(state, wake, clock.now())
    assert entry.state is WakeState.CLOSED
    assert entry.closure.result.policy_row is PolicyRow.ROW_2C
    assert outbox_reason_for(ResultKind.DECISION_ISSUED, PolicyRow.ROW_2C) is OutboxReason.PLAN_SELECTED
    # A Wake-origin Decision starts no ActionIntent (HD-20-4 (a)).
    assert len(fx.adopted(state, type(world.intent))) == 1


def test_a_v2_input_stays_row_2_and_keeps_its_closure_wake(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Under SuccessorPolicy V2 a closed approval stays row 2 (a HOLD), not row
    2c; the Wake is still created (its fingerprint moved) and its review ends
    in a HOLD receipt naming that row, never a Decision."""

    import suzka.r16.decision_issuer as issuer_module
    from suzka.r19.ledger import WakeState

    monkeypatch.setattr(issuer_module, "POLICY_KIND", PolicyKind.SUCCESSOR_POLICY_V2)
    with TestClient(_app(_settings(tmp_path / "v2"))) as client:
        state = client.app.state
        state.r19_scheduler_enabled = False
        clock = fx.Time(state, monkeypatch)
        gid, _decision, intent, _forecast = ip16b._forward(state)
        _only_approval(state)
        ip16b._verdict(state, intent, PolicyVerdict.DENY)
        wake = _scan(state).wake
        assert (wake.cause, wake.policy_kind) == (WakeCause.APPROVAL_CLOSED, PolicyKind.SUCCESSOR_POLICY_V2)
        clock.advance(120)
        claimed = fx.claim(state, wake)
        result = fx.review(state, claimed.claim.claim_id)
        assert result.status is not ReviewStatus.DECISION_ISSUED
        entry = fx.entry(state, wake, clock.now())
        assert entry.state is WakeState.CLOSED
        assert entry.closure.result.policy_row is SuccessorRow.ROW_2
        assert gid


def test_an_older_unclaimed_goal_wake_is_superseded(world) -> None:  # type: ignore[no-untyped-def]
    """GOAL_ADMITTED Wake first, then the closure: the APPROVAL_CLOSED Wake may
    cancel it, and the real terminal accepts it."""

    state, clock = world.state, world.time
    goal_wake = _scan(state).wake
    assert goal_wake.cause is WakeCause.GOAL_ADMITTED
    _only_approval(state)
    _deny(world)
    wake = _scan(state).wake
    assert wake.cause is WakeCause.APPROVAL_CLOSED
    assert may_supersede(goal_wake, wake)
    assert not may_supersede(wake, goal_wake)
    clock.advance(10)
    done = fx.terminal(state, goal_wake, TerminalReason.SUPERSEDED, wake.wakeup_id)
    assert done is not None
    assert fx.entry(state, goal_wake, clock.now()).state.name == "CANCELLED"


# ------------------------------------ §12-1: the stale cause label is pinned


def test_a_skipped_older_candidate_materializes_the_closure_with_its_old_label(world) -> None:  # type: ignore[no-untyped-def]
    """IP-20b interpretation 1, pinned (D1 §12-1, HD-21-6 (a)).  The Goal's
    COMMITMENT_ADMITTED candidate is skipped as SAME_FINGERPRINT while the
    Goal Wake has the same evidence, so it stays un-Woken.  When the approval
    closes, the scan walks candidates oldest cause first: that older candidate
    qualifies at the new fingerprint and its Wake carries ITS label; the
    APPROVAL_CLOSED candidate is then skipped as SAME_FINGERPRINT.  "cause"
    therefore reads "first eligible un-Woken cause of this Goal", not "what
    changed".  If this test ever fails, the label semantics changed and the
    R20 report / research export wording must change with it."""

    state = world.state
    first = _scan(state)
    assert first.wake.cause is WakeCause.GOAL_ADMITTED
    # Nothing else qualifies before the closure; the Commitment candidate sits
    # un-Woken behind the same fingerprint.
    quiet = _scan(state)
    assert (WakeCause.COMMITMENT_ADMITTED, SkipReason.SAME_FINGERPRINT) in _skips(quiet)
    assert (WakeCause.APPROVAL_CLOSED, SkipReason.CAUSE_ALREADY_WOKEN) not in _skips(quiet)
    _deny(world)
    labelled = _scan(state)
    assert labelled.status is MaterializeStatus.COMMITTED
    assert labelled.wake.cause is WakeCause.COMMITMENT_ADMITTED  # the OLD label
    # The closure candidate has nothing left to add: the same evidence has its Wake.
    after = _scan(state)
    assert after.status is MaterializeStatus.NONE_LEFT
    assert (WakeCause.APPROVAL_CLOSED, SkipReason.SAME_FINGERPRINT) in _skips(after)
    assert [wake.cause for wake in _wakes(state)].count(WakeCause.APPROVAL_CLOSED) == 0


def test_with_the_admission_candidates_out_of_the_way_the_closure_labels_itself(world) -> None:  # type: ignore[no-untyped-def]
    """The sibling of the pin: with the earlier causes already woken (or
    superseded) the closure carries its own label."""

    state = world.state
    # Both admission candidates get their Wake at the pre-closure fingerprint
    # (Goal, then Commitment with a later cause); only then the approval closes.
    goal_wake = _scan(state).wake
    commitment_candidate_skip = _scan(state)
    assert (WakeCause.COMMITMENT_ADMITTED, SkipReason.SAME_FINGERPRINT) in _skips(commitment_candidate_skip)
    _only_approval(state)  # the un-Woken Commitment candidate is "superseded" for this case
    _deny(world)
    wake = _scan(state).wake
    assert wake.cause is WakeCause.APPROVAL_CLOSED
    assert wake.evidence_fingerprint != goal_wake.evidence_fingerprint


# --------------------------------------------------------- scope and holds


def test_the_scope_floor_is_the_closure_not_the_admission(world) -> None:  # type: ignore[no-untyped-def]
    """HD-20-1 (a): an Intent admitted long ago but closed just now is in scope;
    one whose closure is older than ``materialize_max_age_s`` is not."""

    state = world.state
    _only_approval(state)
    state.wake_materializer._max_age = timedelta(seconds=600)
    # The window (an hour) ended 5 s ago: the Intent's admission is an hour
    # old, far past the 600 s bound, but its closure is 5 s old.
    _past_window(world)
    result = _scan(state)
    assert result.status is MaterializeStatus.COMMITTED, _skips(result)
    assert result.wake.cause is WakeCause.APPROVAL_CLOSED


def test_a_closure_older_than_the_bound_is_out_of_scope(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    _only_approval(state)
    _deny(world)
    state.wake_materializer._max_age = timedelta(seconds=600)
    clock.advance(700)  # the DENY is now 700 s old
    assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.OUT_OF_SCOPE)]
    assert _wakes(state) == []


def test_a_third_expiry_closure_older_than_the_bound_is_out_of_scope(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    _only_approval(state)
    _three_expiries(world)
    state.wake_materializer._max_age = timedelta(seconds=600)
    clock.advance(700)
    assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.OUT_OF_SCOPE)]


def test_a_closure_with_the_window_ended_and_a_third_expiry_uses_the_earlier(world) -> None:  # type: ignore[no-untyped-def]
    """approval_window_end (an hour out) vs the third expiry's Journal time
    (now): the earlier, the expiry, is the closure instant, so with the window
    5 s past and a 600 s bound the closure is an hour old: OUT_OF_SCOPE."""

    state = world.state
    _only_approval(state)
    _three_expiries(world)
    _past_window(world)
    state.wake_materializer._max_age = timedelta(seconds=600)
    assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.OUT_OF_SCOPE)]


def test_a_window_closure_is_out_of_scope_when_its_window_end_is_too_old(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    _only_approval(state)
    _past_window(world)
    state.wake_materializer._max_age = timedelta(seconds=60)
    clock.advance(3600)  # the window ended an hour ago, the bound is a minute
    assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.OUT_OF_SCOPE)]
    assert _wakes(state) == []


def test_an_archived_closure_event_or_admission_is_out_of_scope(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    _only_approval(state)
    _deny(world)
    real = materializer_module.locate_journal_source
    stage = materializer_module.JournalSourceStage

    def archived_verdict(inspection, event_id, sequence):  # type: ignore[no-untyped-def]
        found = real(inspection, event_id, sequence)
        return (stage.MISSING, ()) if event_id == deny_event else found

    deny_event = next(
        item.event_id
        for item in SubjectHistoryReader(state.event_journal, state.subject_history_store)
        .adopted_records()
        .records
        if type(item.record).__name__ == "PolicyVerdictV1"
    )
    world.monkeypatch.setattr(materializer_module, "locate_journal_source", archived_verdict)
    assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.OUT_OF_SCOPE)]
    world.monkeypatch.setattr(
        materializer_module,
        "locate_journal_source",
        lambda inspection, _e, _s: (stage.MISSING, ()),
    )
    assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.OUT_OF_SCOPE)]
    assert _wakes(state) == []


def test_a_clock_below_the_journal_floor_creates_no_wake(world) -> None:  # type: ignore[no-untyped-def]
    """The window closure is derived from the clock, so an unverifiable clock
    must not create it (D1 §7-3 "Time")."""

    state, clock = world.state, world.time
    _only_approval(state)
    _past_window(world)
    with world.monkeypatch.context() as patch:
        patch.setattr(
            materializer_module,
            "journal_clock_floor",
            lambda records, **_kw: clock.now() + timedelta(days=1),
        )
        assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.CLOCK_UNVERIFIABLE)]
    assert _wakes(state) == []
    assert _scan(state).status is MaterializeStatus.COMMITTED


def test_a_closure_whose_deadline_leaves_no_room_is_not_schedulable(world) -> None:  # type: ignore[no-untyped-def]
    """``wake_window`` returns None when expires − margin is already past: the
    deadline is not silently lost, no Wake is made."""

    state = world.state
    _only_approval(state)
    _deny(world)
    world.monkeypatch.setattr(materializer_module, "wake_window", lambda *a, **k: None)
    assert _skips(_scan(state)) == [(WakeCause.APPROVAL_CLOSED, SkipReason.WAKE_NOT_SCHEDULABLE)]
    assert _wakes(state) == []


def test_disabled_r19_creates_no_approval_wake(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", r19=False))) as client:
        state = client.app.state
        gid, _d, intent, _f = ip16b._forward(state)
        ip16b._verdict(state, intent, PolicyVerdict.DENY)
        result = _scan(state)
        assert result.status is MaterializeStatus.HOLD
        assert fx.operations(state, K.R19_WAKE_MATERIALIZE) == []
        assert gid


# --------------------------------------------- contract: bytes, ref, window


def test_the_intent_ref_kind_is_the_r18_one_and_the_enum_only_grew() -> None:
    assert INTENT_REF_KIND == R18_INTENT_REF_KIND == "r18-intent"
    assert [cause.value for cause in WakeCause] == [
        "GOAL_ADMITTED",
        "COMMITMENT_ADMITTED",
        "FEEDBACK_ADMITTED",
        "OUTCOME_VERIFIED_NO_FEEDBACK",
        "APPROVAL_CLOSED",
    ]
    assert set(contracts_module._CAUSE_REF_KIND) == set(WakeCause)


def _approval_wake(**changes: Any) -> WakeV1:
    base = ip13b._wake(WakeCause.GOAL_ADMITTED)
    values: dict[str, Any] = {
        "cause": WakeCause.APPROVAL_CLOSED,
        "cause_ref": PolicyRef(R18_INTENT_REF_KIND, "r18-intent-1", 1, ip13b.H),
    }
    values.update(changes)
    return replace(base, **values)


def test_approval_wake_bytes_round_trip_and_old_causes_are_unchanged() -> None:
    wake = _approval_wake(origin_refs=(ip13b.COMMITMENT,))
    assert WakeV1.from_json(wake.canonical_bytes()) == wake
    assert b'"cause":"APPROVAL_CLOSED"' in wake.canonical_bytes()
    # Pre-IP-21d causes encode and decode exactly as before.
    for cause in (
        WakeCause.GOAL_ADMITTED,
        WakeCause.COMMITMENT_ADMITTED,
        WakeCause.FEEDBACK_ADMITTED,
        WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK,
    ):
        old = ip13b._wake(cause)
        assert WakeV1.from_json(old.canonical_bytes()) == old
        assert f'"cause":"{cause.value}"'.encode() in old.canonical_bytes()
    # The ref kind is fixed, and an Outcome ref is refused for this cause.
    with pytest.raises(ValueError, match="r18-intent"):
        _approval_wake(cause_ref=ip13b.CAUSE_REF[WakeCause.GOAL_ADMITTED])
    with pytest.raises(ValueError, match="Outcome"):
        _approval_wake(origin_refs=(ip13b.COMMITMENT, ip13b.OUTCOME))


def test_may_supersede_extends_only_to_admission_wakes() -> None:
    new = _approval_wake()
    goal = ip13b._wake(WakeCause.GOAL_ADMITTED, evidence_fingerprint="d" * 64)
    commitment = ip13b._wake(WakeCause.COMMITMENT_ADMITTED, evidence_fingerprint="e" * 64)
    assert may_supersede(goal, new)
    # A Commitment Wake is cancelled only when the new Wake cites the same Commitment.
    assert not may_supersede(commitment, new)
    assert may_supersede(commitment, _approval_wake(origin_refs=(ip13b.COMMITMENT,)))
    # Outcome-cause Wakes are never cancelled by it (another Intent, its own evidence).
    for cause in (WakeCause.FEEDBACK_ADMITTED, WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK):
        assert not may_supersede(ip13b._wake(cause, evidence_fingerprint="c" * 64), new)
    # An APPROVAL_CLOSED Wake is not cancelled by an admission Wake, nor by another closure.
    assert not may_supersede(new, goal)
    assert not may_supersede(new, _approval_wake(evidence_fingerprint="b" * 64))
    # The IP-20 conditions still apply: same Goal revision, Context, window.
    for changes in (
        {"goal_revision": 2},
        {"goal_id": "goal-2"},
        {"context_id": "context-2"},
        {"not_before_utc": "2026-11-24T00:00:00Z", "expires_at_utc": "2026-11-25T00:00:00Z"},
        {"incarnation_id": "c" * 64},
    ):
        assert not may_supersede(goal, replace(new, **changes)), changes
    # The pre-IP-21d rules are untouched.
    assert may_supersede(ip13b._wake(WakeCause.GOAL_ADMITTED), ip13b._wake(WakeCause.FEEDBACK_ADMITTED, evidence_fingerprint="a" * 64)) is True
    assert not may_supersede(ip13b._wake(WakeCause.FEEDBACK_ADMITTED), goal)


def _window(cause: WakeCause, *, caused: str, created: str, goal_deadline: str | None = None, commitment: str | None = None):  # type: ignore[no-untyped-def]
    return wake_window(
        cause,
        cause_utc=caused,
        created_utc=created,
        goal_deadline_utc=goal_deadline,
        commitment_deadline_utc=commitment,
        timing=WakeTiming(),
    )


def test_wake_window_rows_for_approval_closed() -> None:
    created = "2026-11-20T12:00:00Z"
    early = "2026-11-19T12:00:00Z"  # the Intent's admission, a day before
    # Creation + 60 s, expiring in 7 days; the (older) cause time does not matter.
    assert _window(WakeCause.APPROVAL_CLOSED, caused=early, created=created) == (
        "2026-11-20T12:01:00Z",
        "2026-11-27T12:00:00Z",
    )
    assert _window(WakeCause.APPROVAL_CLOSED, caused=created, created=created) == _window(
        WakeCause.APPROVAL_CLOSED, caused=early, created=created
    )
    # Same numbers as the FEEDBACK branch.
    assert _window(WakeCause.APPROVAL_CLOSED, caused=early, created=created) == _window(
        WakeCause.FEEDBACK_ADMITTED, caused=early, created=created
    )
    # The 60 s delay is capped at expires - margin (600 s): a deadline 650 s out.
    assert _window(
        WakeCause.APPROVAL_CLOSED, caused=early, created=created, commitment="2026-11-20T12:10:50Z"
    ) == ("2026-11-20T12:00:50Z", "2026-11-20T12:10:50Z")
    # expires - margin already past: no Wake, even though the cause is old.
    assert (
        _window(WakeCause.APPROVAL_CLOSED, caused=early, created=created, goal_deadline="2026-11-20T12:05:00Z")
        is None
    )
    # ... where GOAL_ADMITTED keeps its literal rule (not_before, not expires - margin).
    assert (
        _window(WakeCause.APPROVAL_CLOSED, caused=early, created=created, goal_deadline="2026-11-20T12:05:00Z")
        == _window(WakeCause.FEEDBACK_ADMITTED, caused=early, created=created, goal_deadline="2026-11-20T12:05:00Z")
    )
    # A cause after its creation is refused.
    with pytest.raises(ValueError):
        _window(WakeCause.APPROVAL_CLOSED, caused="2026-11-20T12:00:01Z", created=created)


# ------------------------------------------- fingerprint: V2 and V3 (D1 §7-2)


@pytest.mark.parametrize("version", (PolicyKind.SUCCESSOR_POLICY_V2, PolicyKind.SUCCESSOR_POLICY_V3))
@pytest.mark.parametrize("closed", (LedgerState.POLICY_DENIED, LedgerState.APPROVAL_EXPIRED_UNCLAIMED))
def test_closing_an_intent_changes_the_fingerprint_for_v2_and_v3(version, closed) -> None:  # type: ignore[no-untyped-def]
    """SAME_FINGERPRINT would silently drop the cause if it did not."""

    from suzka.r19.contracts import wake_fingerprint

    ledger_ref = PolicyRef(R18_INTENT_REF_KIND, "r18-intent-" + "1" * 8, 1, "a" * 64)
    verdict_ref = PolicyRef("r18-verdict", "v" * 8, 1, "b" * 64)
    open_input = ip8b._active(
        version=version,
        predecessor=ip8b.PredecessorState.QUALIFIED,
        predecessor_ref=ip8b.PREDECESSOR,
        ledger=LedgerState.INTENT_ISSUED,
        ledger_refs=(ledger_ref,),
    )
    closed_input = replace(
        open_input, ledger=closed, ledger_refs=tuple(sorted((ledger_ref, verdict_ref), key=lambda r: r.sort_key))
    )
    opened, shut = wake_fingerprint(open_input), wake_fingerprint(closed_input)
    assert opened is not None and shut is not None and opened != shut
    # The fingerprint is in the input's own version domain.
    assert open_input.version is version
    # The ledger state alone moves it (same refs).
    assert wake_fingerprint(replace(open_input, ledger=closed)) != opened
    # Row: V3 reads a closed approval as row 2c, V2 keeps row 2 (IP-17 §12-2).
    expected = (
        SuccessorRow.ROW_2C if version is PolicyKind.SUCCESSOR_POLICY_V3 else SuccessorRow.ROW_2
    )
    assert evaluate(closed_input).row is expected


# ------------------------------- an older build and an unknown cause (D1 §15)


def test_an_older_reader_blocks_the_incarnation_instead_of_failing(world) -> None:  # type: ignore[no-untyped-def]
    """D1 §15 / §2: a store written by IP-21d code holds an APPROVAL_CLOSED Wake;
    a pre-IP-21d build cannot decode it.  Simulated by an enum that does not
    know the value: the reader maps the decode error to a ``DIVERGED`` block of
    the whole incarnation (``classify`` catches ``ValueError``), never an
    exception.  This marks the whole incarnation's SubjectHistory DIVERGED (not
    only R19).  NOT verified here: an owner HOLD (R16 / R17 / R18), startup of
    an older build, or an actual older binary."""

    state = world.state
    wake = _closure_wake(world, _deny)
    assert wake.cause is WakeCause.APPROVAL_CLOSED
    real = contracts_module.parse_enum

    def older(value, enum, name):  # type: ignore[no-untyped-def]
        if enum is WakeCause and value == "APPROVAL_CLOSED":
            raise ValueError("unknown cause")
        return real(value, enum, name)

    world.monkeypatch.setattr(contracts_module, "parse_enum", older)
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert (found.blocked, found.records) == (HeadBlockReason.DIVERGED, ())
