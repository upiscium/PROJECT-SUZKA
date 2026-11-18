"""IP-17c: the EXECUTION_CLAIM handler and the confirm flow (#411 D1 r7 §1, §3b,
§6, §7, §8a, §10 N3 / N4 / N5 / N6 / N8).

The CLAIM is nonce consumption + the claim in ONE R07 event.  It causes no
effect: the executor is IP-18, so the target stays byte-identical.  The API,
its guards and the page are in ``test_r18_ip17c_api.py``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16.history_records import (
    HistoryOperationKind,
    HistoryRecordOperationV1,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError
from suzka.r16.successor_policy import LedgerState, SuccessorRow
from suzka.r18.admission import AdmissionGate
from suzka.r18.approval_owner import (
    ApprovalStatus,
    R18ApprovalReason,
    ReviewStatus,
)
from suzka.r18.contracts import (
    ApprovalChallengeV1,
    ApprovalChannel,
    Direction,
    ExecutionClaimV1,
    PolicyVerdictV1,
)
from suzka.r18.ledger import RecordAt, derive_ledger
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import EventLifecycle

import test_r16_ip10_decision_issuer as ip10
import test_r16_ip16b_ledger_reader as ip16b
import test_r18_ip16a2_intent_issuer as ip16a2
import test_r18_ip17a_policy as ip17a
import test_r18_ip17b_approval as ip17b
from test_r16_u2a_source_owner import _app

K = HistoryOperationKind
_settings = ip16a2._settings
_allowed = ip17b._allowed
_Clock = ip17b._Clock
_adopted = ip17b._adopted
_approval_rows = ip17b._approval_rows
_fake_timers = ip17b._fake_timers
_no_failed_events = ip16a2._no_failed_events
_assert_row_2c = ip17a._assert_row_2c

CLAIM_KINDS = (K.R18_EXECUTION_CLAIM,)
_APPROVAL_EVENTS = frozenset(
    {
        AgentEventType.R18_CHALLENGE_ISSUE,
        AgentEventType.R18_APPROVAL_EXPIRE,
        AgentEventType.R18_EXECUTION_CLAIM,
    }
)


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip17c"))


# ----------------------------------------------------------------- helpers


def _claim_rows(state):  # type: ignore[no-untyped-def]
    return _approval_rows(state, *CLAIM_KINDS)


def _claim_events(state) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type is AgentEventType.R18_EXECUTION_CLAIM
        and row.lifecycle is EventLifecycle.ACCEPTED
    )


def _all_events(state) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type in _APPROVAL_EVENTS and row.lifecycle is EventLifecycle.ACCEPTED
    )


def _consumptions(state):  # type: ignore[no-untyped-def]
    return state.subject_history_store._read(
        "SELECT nonce, challenge_digest, claim_id FROM approval_nonce_consumptions"
    )


def _target_bytes(state) -> str:  # type: ignore[no-untyped-def]
    path = state.settings.r18.target_path
    return sha256(Path(path).read_bytes()).hexdigest()


def _issued(state, intent_id):  # type: ignore[no-untyped-def]
    """review -> issue only (the challenge is written, not claimed)."""

    review = state.approval_owner.review(intent_id)
    assert review.challenge is not None, review
    issued = state.approval_owner.issue(
        intent_id, review.challenge.nonce, review.challenge.confirmation_code
    )
    assert issued.status is ApprovalStatus.ISSUED, issued
    return review.challenge


def _confirm(state, intent_id):  # type: ignore[no-untyped-def]
    review = state.approval_owner.review(intent_id)
    assert review.challenge is not None, review
    return review.challenge, state.approval_owner.confirm(
        intent_id, review.challenge.nonce, review.challenge.confirmation_code
    )


def _ledger(state, intent) -> LedgerState:  # type: ignore[no-untyped-def]
    from suzka.r16.history_reader import SubjectHistoryReader

    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    return derive_ledger(
        (RecordAt(item.record, item.processing_sequence) for item in found.records),
        intent.obligation_group_key,
        state.approval_owner._clock(),
    ).state


class _ClosedGate(AdmissionGate):
    def is_open(self) -> bool:
        return False


# ============================================================ the CLAIM


def test_confirm_issues_then_claims_in_one_event_and_causes_no_effect(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        timers = _fake_timers(state)
        clock = _Clock(state)
        gid, intent, _forecast = _allowed(state)
        target_before = _target_bytes(state)
        review = owner.review(intent.intent_id)
        challenge = review.challenge
        assert challenge is not None
        done = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (done.status, done.attempt) == (ApprovalStatus.CLAIMED, 1)
        claim = done.claim
        assert claim is not None and done.challenge == challenge
        # One challenge row and one claim row; the claim is ONE R07 event that
        # carries the consumption and the claim together (§6-1 step 8).
        assert _adopted(state, ApprovalChallengeV1) == [challenge]
        assert _adopted(state, ExecutionClaimV1) == [claim]
        rows = _claim_rows(state)
        assert [(row.operation_kind, row.phase) for row in rows] == [
            (K.R18_EXECUTION_CLAIM, HistoryPhase.FINALIZED)
        ]
        assert _claim_events(state) == 1
        assert _consumptions(state) == [(challenge.nonce, challenge.challenge_digest, claim.claim_id)]
        operation = HistoryRecordOperationV1.from_bytes(rows[0].pending_bytes)  # type: ignore[arg-type]
        assert operation.record == claim and operation.event_id == claim.claim_event_id
        assert operation.processing_sequence == claim.claim_sequence
        # The claim's fields (§6-1 step 7, §12-1).
        assert claim.approval_channel is ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1
        assert claim.runtime_instance_id == owner.runtime_instance_id
        assert claim.runtime_instance_id != owner.boot_id and claim.boot_id == owner.boot_id
        assert claim.commitment_deadline_utc == challenge.commitment_deadline_utc
        assert (claim.intent_id, claim.intent_digest, claim.operation_key, claim.direction) == (
            intent.intent_id,
            intent.intent_digest,
            intent.forward_key,
            Direction.FORWARD,
        )
        assert claim.claimed_utc == clock.now.strftime(ip17b.UTC_FORMAT)
        # invalidate: what a review showed is forgotten, the pending challenge
        # dropped and the expiry timer cancelled.
        assert intent.intent_id not in owner._shown
        assert intent.intent_id not in owner._reviewed
        assert owner.pending_challenge(intent.intent_id) is None
        assert all(timer.cancelled for timer in timers)
        # CLAIMED stays row 2 until IP-18's sweep (§7); the target is untouched.
        assert _ledger(state, intent) is LedgerState.CLAIMED
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
        assert _target_bytes(state) == target_before
        assert owner.review(intent.intent_id).status is ReviewStatus.HOLD
        _no_failed_events(state)


def test_n3_a_second_confirm_after_a_claim_is_refused_and_one_claim_exists(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge, done = _confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        events = _all_events(state)
        again = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (again.status, again.reason) == (
            ApprovalStatus.HOLD,
            R18ApprovalReason.CHALLENGE_NOT_LIVE,
        )
        # Straight to the CLAIM event: the nonce is consumed.
        direct = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (direct.status, direct.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.NONCE_USED)
        assert len(_adopted(state, ExecutionClaimV1)) == 1 and len(_consumptions(state)) == 1
        assert len(_claim_rows(state)) == 1
        assert _all_events(state) == events + 1  # only the direct claim event
        _no_failed_events(state)


def test_a_confirm_for_a_written_nonce_goes_straight_to_the_claim(app) -> None:  # type: ignore[no-untyped-def]
    """r6 P1-1: confirm -> challenge written -> CLAIM HOLD -> retry within the
    TTL: the retry goes straight to the CLAIM; one challenge row, one attempt."""

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        owner._gate = _ClosedGate()
        held = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (held.status, held.reason) == (
            ApprovalStatus.HOLD,
            R18ApprovalReason.ADMISSION_GATE_CLOSED,
        )
        assert _claim_rows(state) == [] and _consumptions(state) == []
        events = _all_events(state)
        owner._gate = ip17b.ip17a.ALWAYS_OPEN_UNTIL_IP18
        done = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (done.status, done.attempt) == (ApprovalStatus.CLAIMED, 1)
        # Exactly one challenge row, one claim: the retry wrote no second
        # challenge, and issue() submitted no event for the written nonce.
        assert len(_approval_rows(state, K.R18_CHALLENGE_ISSUE)) == 1
        assert _all_events(state) == events + 1
        _no_failed_events(state)


def test_a_wrong_code_or_unknown_nonce_consumes_nothing(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        events = _all_events(state)
        wrong = owner.confirm(intent.intent_id, challenge.nonce, "WRONG000")
        assert wrong.reason is R18ApprovalReason.CODE_MISMATCH
        unknown = owner.confirm(intent.intent_id, "0" * 64, challenge.confirmation_code)
        assert unknown.reason is R18ApprovalReason.UNKNOWN_CHALLENGE
        # The CLAIM event itself checks the code and the nonce again.
        for code, nonce, reason in (
            ("WRONG000", challenge.nonce, R18ApprovalReason.CODE_MISMATCH),
            (challenge.confirmation_code, "0" * 64, R18ApprovalReason.UNKNOWN_CHALLENGE),
        ):
            held = owner._submit_claim(intent.intent_id, nonce, code).result(timeout=30).value
            assert (held.status, held.reason) == (ApprovalStatus.HOLD, reason)
        assert _claim_rows(state) == [] and _consumptions(state) == []
        assert owner._pending is None
        assert _all_events(state) == events + 2
        _no_failed_events(state)


# ============================================================ N4: expiry


def test_n4_ttl_and_window_expiry_refuse_the_claim(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        clock.advance(300)  # the TTL has passed
        held = owner._submit_claim(
            intent.intent_id, challenge.nonce, challenge.confirmation_code
        ).result(timeout=30).value
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.CHALLENGE_NOT_LIVE)
        clock.now = intent.approval_window_end  # the window has ended
        held = owner._submit_claim(
            intent.intent_id, challenge.nonce, challenge.confirmation_code
        ).result(timeout=30).value
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.WINDOW_CLOSED)
        assert _claim_rows(state) == [] and _consumptions(state) == []
        _no_failed_events(state)


def test_n4_expire_then_a_claim_with_the_clock_set_back_is_refused(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        start = clock.now
        clock.advance(301)
        assert ip17b._expire(state, challenge).expiration is not None
        clock.now = start  # back inside the TTL
        assert challenge.is_live(start.strftime(ip17b.UTC_FORMAT), owner.boot_id)
        held = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.CHALLENGE_NOT_LIVE)
        assert _claim_rows(state) == [] and _consumptions(state) == []
        _no_failed_events(state)


def test_n4_expire_versus_claim_exactly_one_wins(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # CLAIM first: the expiry is then ALREADY_CONSUMED (and nothing is written).
    with TestClient(_app(_settings(tmp_path / "claim-first"))) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge, done = _confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        clock.advance(301)
        ended = ip17b._expire(state, challenge)
        assert (ended.status, ended.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.ALREADY_CONSUMED)
        assert len(_adopted(state, ExecutionClaimV1)) == 1
        assert _approval_rows(state, K.R18_APPROVAL_EXPIRE) == []
    # EXPIRE first (BOOT_CHANGED: the challenge's boot is not this process's):
    # the CLAIM is then refused and nothing is consumed.
    with TestClient(_app(_settings(tmp_path / "expire-first"))) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        owner._boot_id = "00000000-0000-4000-8000-000000000001"
        assert ip17b._expire(state, challenge).status is ApprovalStatus.EXPIRED
        held = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.CHALLENGE_NOT_LIVE)
        assert _adopted(state, ExecutionClaimV1) == [] and _consumptions(state) == []


# ====================================================== N5: digest mismatch


def _fixture_challenge(state, intent, forecast, **changes: Any) -> ApprovalChallengeV1:  # type: ignore[no-untyped-def]
    """A challenge with one bound field mutated, committed as a FIXTURE through
    the real R07 runtime (the store checks only intent id, direction and key)."""

    owner = state.approval_owner
    now = owner._clock().replace(microsecond=0)
    base = ApprovalChallengeV1.for_forward(
        intent,
        forecast.forecast_digest,
        _verdict(state, intent),
        nonce="ab" * 32,
        issued_utc=now.strftime(ip17b.UTC_FORMAT),
        expires_utc=(now + timedelta(seconds=120)).strftime(ip17b.UTC_FORMAT),
        boot_id=owner.boot_id,
        commitment_deadline_utc=None,
    )
    record = replace(base, **changes)
    ip16b._commit_record(
        state,
        lambda event: ip16b._operation(
            state, event, K.R18_CHALLENGE_ISSUE, ip16b.R.APPROVAL_CHALLENGE, record, 1
        ),
    )
    return record


def _verdict(state, intent):  # type: ignore[no-untyped-def]
    found = _adopted(state, PolicyVerdictV1)
    return next(item for item in found if item.intent_id == intent.intent_id)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("intent_digest", "11" * 32, R18ApprovalReason.CHALLENGE_STALE),
        ("payload_digest", "22" * 32, R18ApprovalReason.CHALLENGE_STALE),
        ("expectation_digest", "33" * 32, R18ApprovalReason.CHALLENGE_STALE),
        ("target_instance_id", "44444444-4444-4444-8444-444444444444", R18ApprovalReason.CHALLENGE_STALE),
        ("target_expected_revision", 7, R18ApprovalReason.CHALLENGE_STALE),
        ("boot_id", "55555555-5555-4555-8555-555555555555", R18ApprovalReason.CHALLENGE_NOT_LIVE),
        ("commitment_deadline_utc", "2999-01-01T00:00:00Z", R18ApprovalReason.CHALLENGE_STALE),
    ],
)
def test_n5_each_bound_field_mutated_holds_the_claim(tmp_path, field, value, reason) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / f"n5-{field}"))) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, forecast = _allowed(state)
        record = _fixture_challenge(state, intent, forecast, **{field: value})
        held = owner.claim(intent.intent_id, record.nonce, record.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, reason)
        assert _claim_rows(state) == [] and _consumptions(state) == []
        assert owner._pending is None
        _no_failed_events(state)


def test_n5_a_challenge_outliving_the_window_or_the_ttl_holds_the_claim(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "n5-window"))) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        _gid, intent, forecast = _allowed(state)
        # Expires after the window end (the constructor allows it; the claim
        # re-checks it, §6-1 step 3).
        late = _fixture_challenge(
            state,
            intent,
            forecast,
            issued_utc=(intent.approval_window_end - timedelta(seconds=30)).strftime(ip17b.UTC_FORMAT),
            expires_utc=(intent.approval_window_end + timedelta(seconds=30)).strftime(ip17b.UTC_FORMAT),
        )
        clock.now = intent.approval_window_end - timedelta(seconds=10)
        held = owner.claim(intent.intent_id, late.nonce, late.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.CHALLENGE_STALE)
        assert _claim_rows(state) == []


# ============================================================ N8 and gates


def test_n8_a_closed_admission_gate_holds_the_claim(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        owner._gate = _ClosedGate()
        held = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.ADMISSION_GATE_CLOSED)
        assert _claim_rows(state) == [] and owner._pending is None
        _no_failed_events(state)


def test_drift_after_the_challenge_holds_the_claim_and_closes_at_the_window(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.review_value import REVIEW_VALUE_ID

    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", ip16a2.TOKEN)
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        for action in ("freeze", "unfreeze"):
            assert client.post(f"/api/values/{REVIEW_VALUE_ID}/{action}", headers=ip16a2.ADMIN).status_code == 200
        held = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.SOURCE_NOT_CURRENT)
        assert _claim_rows(state) == [] and _consumptions(state) == []
        events = _all_events(state)
        # No Human action: the window closes it (§8a: CLAIM HOLD -> window end).
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
        _assert_row_2c(ip17a._closed_preview(state, gid, intent, monkeypatch))
        assert _all_events(state) == events
        _no_failed_events(state)


def test_a_claim_after_a_successor_2c_decision_is_refused_even_with_the_clock_set_back(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        _assert_row_2c(ip17a._closed_preview(state, gid, intent, monkeypatch))
        assert intent.window_open(owner._clock())  # the clock is back inside
        for held in (
            owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code),
            owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code),
        ):
            assert held.status is ApprovalStatus.HOLD
            assert held.reason in (
                R18ApprovalReason.SOURCE_NOT_CURRENT,
                R18ApprovalReason.LEDGER_UNSAFE,
            )
        assert _claim_rows(state) == [] and _consumptions(state) == []
        _no_failed_events(state)


def test_r18_off_the_claim_holds_before_any_read(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.history_reader import SubjectHistoryReader

    with TestClient(_app(_settings(tmp_path / "off", r18=False))) as client:
        state = client.app.state
        owner = state.approval_owner
        assert not owner.enabled

        def no_read(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("read while r18 is disabled")

        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "classify", no_read)
            patch.setattr(state._plan_preview_owner, "_state_snapshot", no_read)
            results = [
                owner._submit_claim("intent-" + "0" * 64, "0" * 64, "AAAAAAAA").result(timeout=30).value,
                owner.claim("intent-" + "0" * 64, "0" * 64, "AAAAAAAA"),
                owner.confirm("intent-" + "0" * 64, "0" * 64, "AAAAAAAA"),
            ]
            for held in results:
                assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.R18_DISABLED)
            assert owner.status("intent-" + "0" * 64).reason is R18ApprovalReason.R18_DISABLED
            assert owner.open_approvals().reason is R18ApprovalReason.R18_DISABLED
        assert state.subject_history_store.has_record_operations() is False
        _no_failed_events(state)


def test_an_inverse_intent_is_deferred_at_the_claim(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        owner = client.app.state.approval_owner
        inverse = "inverse-intent-" + "0" * 64
        for held in (
            owner.claim(inverse, "0" * 64, "AAAAAAAA"),
            owner.confirm(inverse, "0" * 64, "AAAAAAAA"),
        ):
            assert held.reason is R18ApprovalReason.INVERSE_DEFERRED
        assert _claim_rows(client.app.state) == []


def test_the_soft_band_refuses_the_claim() -> None:
    from suzka.runtime.event_journal import (
        SOFT_BAND_ADMITTED_EVENT_TYPES,
        EventJournalAdmissionBand,
        EventJournalAdmissionStatus,
    )

    assert AgentEventType.R18_EXECUTION_CLAIM not in SOFT_BAND_ADMITTED_EVENT_TYPES
    soft = EventJournalAdmissionStatus(
        True, "journal_soft_limit", 0, 1, 0, 1, True, False, False,
        band=EventJournalAdmissionBand.SOFT,
    )
    assert not soft.admits(AgentEventType.R18_EXECUTION_CLAIM)
    assert not soft.admits(AgentEventType.R18_CHALLENGE_ISSUE)
    assert soft.admits(AgentEventType.R18_APPROVAL_EXPIRE)
    assert soft.admits(AgentEventType.PLAN_PREVIEW_REVIEW)


def test_confirm_never_raises_on_a_refused_band_a_stopped_runtime_or_a_timeout(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from concurrent.futures import Future, TimeoutError as FutureTimeout

    from suzka.runtime.agent_runtime import AgentRuntimeStopped
    from suzka.runtime.event_journal import EventJournalError

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        args = (intent.intent_id, challenge.nonce, challenge.confirmation_code)

        def refused(*_a):  # type: ignore[no-untyped-def]
            future: Future[Any] = Future()
            future.set_exception(EventJournalError("journal_soft_limit"))
            return future

        def stopped(*_a):  # type: ignore[no-untyped-def]
            raise AgentRuntimeStopped(None)  # type: ignore[arg-type]

        for submit in (refused, stopped):
            monkeypatch.setattr(owner, "_submit_claim", submit)
            held = owner.confirm(*args)
            assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.JOURNAL_UNAVAILABLE)
        monkeypatch.setattr(owner, "_submit_claim", lambda *_a: Future())

        def instant_timeout(future):  # type: ignore[no-untyped-def]
            raise FutureTimeout()

        monkeypatch.setattr(owner, "_run", instant_timeout)
        assert owner.confirm(*args).reason is R18ApprovalReason.JOURNAL_UNAVAILABLE
        monkeypatch.undo()
        assert _claim_rows(state) == []
        # The challenge is still there: the real path works.
        assert owner.confirm(*args).status is ApprovalStatus.CLAIMED


# ======================================================= §3b: pre-checks


def test_every_claim_condition_holds_before_prepare(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        store = state.subject_history_store
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        code = challenge.confirmation_code

        def hold(reason: R18ApprovalReason) -> None:
            result = owner._submit_claim(intent.intent_id, challenge.nonce, code).result(timeout=30).value
            assert (result.status, result.reason) == (ApprovalStatus.HOLD, reason)
            assert _claim_rows(state) == [] and _consumptions(state) == []
            assert owner._pending is None

        # A reserved but unadopted ``challenge-end:`` / ``claim:`` key
        # (nonce / operation_key / challenge_digest UNIQUE) is IN_FLIGHT.
        real_keys = store.live_keys
        for prefix in ("challenge-end:", "claim:"):
            with monkeypatch.context() as patch:
                patch.setattr(
                    store,
                    "live_keys",
                    lambda keys, prefix=prefix: frozenset(k for k in keys if k.startswith(prefix))
                    | real_keys(keys),
                )
                hold(R18ApprovalReason.IN_FLIGHT)
        # A claim of another incarnation holds the ``claim:`` key (§3b: no
        # claim for the operation key in any incarnation).
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "live_key_owner",
                lambda key: ("0" * 64, "0" * 36) if key.startswith("claim:") else None,
            )
            hold(R18ApprovalReason.LEDGER_UNSAFE)

        def full(length):  # type: ignore[no-untyped-def]
            raise HistoryStoreCapacityError("full")

        with monkeypatch.context() as patch:
            patch.setattr(store, "check_headroom", full)
            hold(R18ApprovalReason.HISTORY_CAPACITY)
        # An unadopted claim row of this Intent (PREPARED by another event).
        owner_pending = owner._in_flight
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_in_flight", lambda *a, **k: True)
            hold(R18ApprovalReason.IN_FLIGHT)
        assert owner._in_flight == owner_pending
        # The real path still claims.
        assert owner.claim(intent.intent_id, challenge.nonce, code).status is ApprovalStatus.CLAIMED
        _no_failed_events(state)


def test_validate_commit_sees_exactly_its_own_prepared_claim_row(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
        seen: list[str] = []
        real = owner._validate_commit

        def spy(event):  # type: ignore[no-untyped-def]
            if owner._pending is not None and event.event_type is AgentEventType.R18_EXECUTION_CLAIM:
                seen.extend(row.phase.value for row in _claim_rows(state))
            return real(event)

        owner._validate_commit = spy  # type: ignore[method-assign]
        try:
            done = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        finally:
            owner._validate_commit = real  # type: ignore[method-assign]
        assert done.status is ApprovalStatus.CLAIMED
        assert seen == [HistoryPhase.PREPARED.value]
        assert owner._pending is None


# ============================================================ N6: crashes


def test_n6_after_a_claim_a_restart_is_still_claimed_with_no_second_claim(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "n6")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _fake_timers(state)
        gid, intent, _forecast = _allowed(state)
        challenge, done = _confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        claim = done.claim
    with TestClient(_app(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        # CLAIMED survives the restart: row 2, one claim, nothing expired.
        assert _adopted(state, ExecutionClaimV1) == [claim]
        assert _ledger(state, intent) is LedgerState.CLAIMED
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
        # The old nonce is refused (another boot, and consumed); no second claim.
        again = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert again.status is ApprovalStatus.HOLD
        direct = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert direct.status is ApprovalStatus.HOLD
        assert len(_adopted(state, ExecutionClaimV1)) == 1 and len(_consumptions(state)) == 1
        assert _approval_rows(state, K.R18_APPROVAL_EXPIRE) == []
        assert state.event_journal.inspect().open_transactions == ()


def test_a_crash_before_the_claim_wal_is_tombstoned_and_a_new_boot_needs_a_new_attempt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime.agent_runtime import AgentRuntimeStopped

    settings = _settings(tmp_path / "crash")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R18_EXECUTION_CLAIM:
                raise RuntimeError("simulated pre-WAL failure")

        owner._validate_commit = crash  # type: ignore[method-assign]
        held = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert held.reason is R18ApprovalReason.JOURNAL_UNAVAILABLE
        with pytest.raises(AgentRuntimeStopped):
            owner._submit_claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _claim_rows(state)] == [HistoryPhase.ABORTED]
        assert _adopted(state, ExecutionClaimV1) == [] and _consumptions(state) == []
        # The challenge belongs to the previous boot: refused (BOOT_CHANGED); a
        # new review + confirm inside the window is attempt 2 and can claim.
        held = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert held.status is ApprovalStatus.HOLD
        _reviewed, done = _confirm(state, intent.intent_id)
        assert (done.status, done.attempt) == (ApprovalStatus.CLAIMED, 2)
        _no_failed_events(state)


def test_a_crash_between_the_challenge_write_and_the_claim_is_boot_changed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "between")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _fake_timers(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = _issued(state, intent.intent_id)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        held = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert held.status is ApprovalStatus.HOLD
        assert _adopted(state, ExecutionClaimV1) == []
        _reviewed, done = _confirm(state, intent.intent_id)
        assert (done.status, done.attempt) == (ApprovalStatus.CLAIMED, 2)


# ========================================================== facts: no event


def test_list_and_status_report_ledger_facts_with_no_event_or_write(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        rows = len(ip10._history_transactions(state))
        events = _all_events(state)
        listed = owner.open_approvals()
        assert listed.reason is None and [item.intent_id for item in listed.items] == [intent.intent_id]
        item = listed.items[0]
        assert (item.ledger_state, item.attempts_used, item.live_challenge) == (
            LedgerState.INTENT_ISSUED.value,
            0,
            False,
        )
        assert item.window_end_utc == intent.approval_window_end_utc
        assert item.operation_key == intent.forward_key
        challenge = _issued(state, intent.intent_id)
        owner._facts_cache.clear()  # past the 2 s reuse
        item = owner.status(intent.intent_id).items[0]
        assert (item.attempts_used, item.live_challenge, item.challenge_expires_utc) == (
            1,
            True,
            challenge.expires_utc,
        )
        unknown = owner.status("intent-" + "0" * 64)
        assert unknown.reason is R18ApprovalReason.INTENT_UNKNOWN
        _challenge, done = _confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        owner._facts_cache.clear()
        assert owner.status(intent.intent_id).items[0].ledger_state == LedgerState.CLAIMED.value
        assert owner.open_approvals().items == ()
        # Reads wrote and submitted nothing beyond the issue and the claim.
        assert _all_events(state) == events + 2
        assert len(ip10._history_transactions(state)) == rows + 2
        _no_failed_events(state)


def test_the_facts_reads_are_coalesced(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.history_reader import SubjectHistoryReader

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        reads = []
        real = SubjectHistoryReader.classify

        def counted(self, *a, **k):  # type: ignore[no-untyped-def]
            reads.append(1)
            return real(self, *a, **k)

        monkeypatch.setattr(SubjectHistoryReader, "classify", counted)
        first = owner.open_approvals()
        for _ in range(20):
            assert owner.open_approvals() == first
        assert len(reads) == 1
        # While another read runs, a caller gets the cached result or BUSY.
        owner._facts_cache.clear()
        assert owner._facts_lock.acquire(blocking=False)
        try:
            assert owner.open_approvals().reason is R18ApprovalReason.REVIEW_BUSY
        finally:
            owner._facts_lock.release()
