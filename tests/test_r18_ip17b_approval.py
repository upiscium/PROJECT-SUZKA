"""IP-17b: the R18 Approval owner (#411 D1 r7 §1, §3, §3b, §8a, §12-1).

Challenge reuse / issue, expiry (timer, startup scan, lazy path), the
per-process boot id, the challenge TTL and the §12-1 contract amendments.  The
CLAIM handler, the approval API and the page are IP-17c: where a test needs a
claim it is a FIXTURE committed through the real R07 runtime.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16.decision_issuer import DecisionIssueStatus
from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import (
    MAX_CHALLENGES_PER_INTENT,
    ApprovalExpirationV1,
    ExpirationReason,
    HistoryOperationKind,
    RecordKind,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError
from suzka.r16.successor_policy import LedgerState, SuccessorRow
from suzka.r18.approval_owner import (
    ApprovalChallengeOwnerV1,
    ApprovalStatus,
    R18ApprovalReason,
    ReviewStatus,
)
from suzka.r18.contracts import (
    ApprovalChallengeV1,
    ApprovalChannel,
    Direction,
    ExecutionClaimV1,
)
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import (
    SOFT_BAND_ADMITTED_EVENT_TYPES,
    EventJournalAdmissionBand,
    EventJournalAdmissionStatus,
    EventLifecycle,
)

import test_r16_ip10_decision_issuer as ip10
import test_r16_ip16b_ledger_reader as ip16b
import test_r18_ip13_contracts as ip13
import test_r18_ip16a2_intent_issuer as ip16a2
import test_r18_ip17a_policy as ip17a
from test_r16_u2a_source_owner import _app

UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_forward = ip16b._forward
_settings = ip16a2._settings
_no_failed_events = ip16a2._no_failed_events
CHANNEL = ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1
_APPROVAL_EVENTS = frozenset(
    {AgentEventType.R18_CHALLENGE_ISSUE, AgentEventType.R18_APPROVAL_EXPIRE}
)


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip17b"))


# ----------------------------------------------------------------- helpers


class _Clock:
    """One settable clock for the Approval and Policy owners."""

    def __init__(self, state, start: datetime | None = None) -> None:  # type: ignore[no-untyped-def]
        self.now = start or datetime.now(timezone.utc).replace(microsecond=0)
        state.approval_owner._clock = self
        state.policy_owner._clock = self

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _allowed(state):  # type: ignore[no-untyped-def]
    gid, _decision, intent, forecast = _forward(state)
    assert state.policy_owner._submit_policy(intent.intent_id).result(timeout=30).value.verdict
    return gid, intent, forecast


def _approval_rows(state, *kinds):  # type: ignore[no-untyped-def]
    kinds = kinds or (
        HistoryOperationKind.R18_CHALLENGE_ISSUE,
        HistoryOperationKind.R18_APPROVAL_EXPIRE,
    )
    return [
        row
        for row in state.subject_history_store.operations(ip10._incarnation(state))
        if row.operation_kind in kinds
    ]


def _approval_events(state) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type in _APPROVAL_EVENTS and row.lifecycle is EventLifecycle.ACCEPTED
    )


def _adopted(state, record_type):  # type: ignore[no-untyped-def]
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return [item.record for item in found.records if type(item.record) is record_type]


def _confirm(state, intent_id):  # type: ignore[no-untyped-def]
    """review -> issue with the shown code (what the confirm route does)."""

    review = state.approval_owner.review(intent_id)
    assert review.challenge is not None, review
    return review, state.approval_owner.issue(
        intent_id, review.challenge.nonce, review.challenge.confirmation_code
    )


def _expire(state, challenge):  # type: ignore[no-untyped-def]
    return state.approval_owner._submit_expire(challenge.challenge_digest).result(timeout=30).value


def _assert_row_2c(result) -> None:  # type: ignore[no-untyped-def]
    ip17a._assert_row_2c(result)


class _FakeTimer:
    timers: list[_FakeTimer] = []

    def __init__(self, delay: float, callback) -> None:  # type: ignore[no-untyped-def]
        self.delay = delay
        self.callback = callback
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


def _fake_timers(state) -> list[_FakeTimer]:  # type: ignore[no-untyped-def]
    made: list[_FakeTimer] = []

    def factory(delay, callback):  # type: ignore[no-untyped-def]
        timer = _FakeTimer(delay, callback)
        made.append(timer)
        return timer

    state.approval_owner._timer_factory = factory
    return made


# ======================================================== §12-1 contracts


def _challenge(deadline: str | None = None, **changes: Any) -> ApprovalChallengeV1:
    intent = ip13._intent()
    return ip13._challenge(intent, commitment_deadline_utc=deadline, **changes)


def test_the_commitment_deadline_is_bound_into_the_code() -> None:
    base = _challenge()
    deadline = _challenge("2026-11-16T00:20:00Z")
    assert base.commitment_deadline_utc is None
    assert b'"commitment_deadline_utc":null' in base.canonical_bytes()
    assert b'"commitment_deadline_utc":"2026-11-16T00:20:00Z"' in deadline.canonical_bytes()
    assert base.challenge_digest != deadline.challenge_digest
    assert base.confirmation_code != deadline.confirmation_code
    for value in (base, deadline):
        assert ApprovalChallengeV1.from_json(value.canonical_bytes()) == value
    for bad in ("2026-11-16T00:20:00", "tomorrow", 5):
        with pytest.raises((ValueError, TypeError)):
            _challenge(bad)  # type: ignore[arg-type]
    # A challenge never outlives the deadline (ISSUED 00:00 / EXPIRES 00:10) ...
    with pytest.raises(ValueError, match="deadline"):
        _challenge("2026-11-16T00:09:59Z")
    # ... nor its Intent's window, and an inverse has no deadline.
    intent = ip13._intent()
    with pytest.raises(ValueError, match="window"):
        ApprovalChallengeV1.for_forward(
            intent,
            ip13.EXPECTATION,
            ip13._allow(intent),
            nonce=ip13.NONCE,
            issued_utc=ip13.ISSUED,
            expires_utc="2026-11-16T01:00:01Z",
            boot_id=ip13.BOOT,
            commitment_deadline_utc=None,
        )
    inverse = ip13._inverse()
    with pytest.raises(ValueError, match="inverse"):
        ApprovalChallengeV1(
            ip13.NONCE, Direction.INVERSE, inverse.intent_id, ip13.H, None, ip13.TARGET, 1,
            ip13.H, ip13.H, ip13.ISSUED, ip13.EXPIRES, ip13.BOOT, "2026-11-16T00:20:00Z",
        )


def test_the_claim_carries_the_deadline_and_a_closed_channel() -> None:
    challenge = _challenge("2026-11-16T00:20:00Z")

    def claim(source: ApprovalChallengeV1, at: str = "2026-11-16T00:05:00Z") -> ExecutionClaimV1:
        return ExecutionClaimV1.claim(
            source,
            runtime_instance_id=ip13.RUNTIME,
            boot_id=ip13.BOOT,
            claim_event_id=ip13.EVENT_C,
            claim_sequence=22,
            claimed_utc=at,
            approval_channel=CHANNEL,
        )

    made = claim(challenge)
    assert made.commitment_deadline_utc == challenge.commitment_deadline_utc
    assert made.approval_channel is ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1
    assert [member.value for member in ApprovalChannel] == ["LOCAL_LOOPBACK_TYPED_CODE_V1"]
    assert b'"approval_channel":"LOCAL_LOOPBACK_TYPED_CODE_V1"' in made.canonical_bytes()
    assert ExecutionClaimV1.from_json(made.canonical_bytes()) == made
    assert claim(_challenge()).claim_digest != made.claim_digest
    with pytest.raises(TypeError):
        replace(made, approval_channel="LOCAL_LOOPBACK_TYPED_CODE_V1")  # type: ignore[arg-type]
    for payload in (
        made.canonical_bytes().replace(b"LOCAL_LOOPBACK_TYPED_CODE_V1", b"REMOTE"),
        made.canonical_bytes().replace(b'"approval_channel":', b'"approvalchannel":'),
    ):
        with pytest.raises((ValueError, TypeError)):
            ExecutionClaimV1.from_json(payload)
    # The deadline is checked at the claim, and the target binds it.
    with pytest.raises(ValueError, match="deadline"):
        replace(made, claimed_utc="2026-11-16T00:20:00Z")
    from suzka.r18.target import TargetBindingError, _bind

    intent = ip13._intent()
    _bind(challenge, made, Direction.FORWARD, intent.intent_id, intent.intent_digest)
    with pytest.raises(TargetBindingError):
        _bind(
            challenge, claim(_challenge()), Direction.FORWARD, intent.intent_id, intent.intent_digest
        )


def test_the_contract_docstring_says_a_restart_voids_every_challenge() -> None:
    doc = ApprovalChallengeV1.__doc__ or ""
    assert "HD-10" in doc and "per API process" in doc and "kernel boot unit" in doc


def test_the_ttl_setting_is_bounded() -> None:
    from pydantic import ValidationError

    from suzka.config.schema import R18Settings

    assert R18Settings().approval_ttl_seconds == 300
    for value in (59, 901, "300", 300.0):
        with pytest.raises(ValidationError):
            R18Settings(approval_ttl_seconds=value)  # type: ignore[arg-type]
    assert R18Settings(approval_ttl_seconds=900).approval_ttl_seconds == 900


# ====================================================== review (in memory)


def test_a_review_writes_nothing_and_returns_the_same_live_challenge(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        assert type(owner) is ApprovalChallengeOwnerV1 and owner.enabled
        clock = _Clock(state)
        _gid, intent, forecast = _allowed(state)
        rows = len(ip10._history_transactions(state))
        events = _approval_events(state)
        first = owner.review(intent.intent_id)
        assert first.status is ReviewStatus.BUILT and first.attempt == 1
        challenge = first.challenge
        assert challenge is not None
        assert (challenge.intent_digest, challenge.operation_key, challenge.payload_digest) == (
            intent.intent_digest,
            intent.forward_key,
            intent.payload_digest,
        )
        assert challenge.expectation_digest == forecast.forecast_digest
        assert challenge.boot_id == owner.boot_id
        assert challenge.issued_utc == clock.now.strftime(UTC_FORMAT)
        assert challenge.expires_utc == (clock.now + timedelta(seconds=300)).strftime(UTC_FORMAT)
        assert first.window_end_utc == intent.approval_window_end_utc
        # Never replaced while live: any number of reviews return the same one.
        for _ in range(5):
            clock.advance(1)
            again = owner.review(intent.intent_id)
            assert (again.status, again.challenge, again.attempt) == (
                ReviewStatus.PENDING,
                challenge,
                1,
            )
        # Nothing was written and no event was submitted (§3 / §4).
        assert len(ip10._history_transactions(state)) == rows
        assert _approval_events(state) == events
        assert _approval_rows(state) == []
        # Dead (TTL passed): dropped, a new one with a fresh nonce.
        clock.advance(300)
        fresh = owner.review(intent.intent_id)
        assert fresh.status is ReviewStatus.BUILT and fresh.challenge != challenge
        assert fresh.challenge is not None and fresh.challenge.nonce != challenge.nonce
        assert owner.pending_challenge(intent.intent_id) == fresh.challenge
        _no_failed_events(state)


def test_the_challenge_expiry_is_capped_by_the_window_end(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        clock = _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        clock.now = intent.approval_window_end - timedelta(seconds=40)
        review = state.approval_owner.review(intent.intent_id)
        assert review.challenge is not None
        assert review.challenge.expires_utc == intent.approval_window_end_utc
        clock.now = intent.approval_window_end - timedelta(seconds=1)
        # One second of window left: still reviewable, expires at the end.
        assert state.approval_owner.review(intent.intent_id).challenge is not None
        clock.now = intent.approval_window_end
        closed = state.approval_owner.review(intent.intent_id)
        assert (closed.status, closed.reason) == (ReviewStatus.HOLD, R18ApprovalReason.WINDOW_CLOSED)


def test_a_review_holds_until_there_is_an_allow(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _gid, _decision, intent, _forecast = _forward(state)
        # No verdict yet, then a DENY.
        pending = owner.review(intent.intent_id)
        assert (pending.status, pending.reason) == (ReviewStatus.HOLD, R18ApprovalReason.POLICY_PENDING)
        assert owner.review("intent-" + "0" * 64).reason is R18ApprovalReason.INTENT_UNKNOWN
        assert owner.review("inverse-intent-" + "0" * 64).reason is R18ApprovalReason.INVERSE_DEFERRED
        assert owner.review("").reason is R18ApprovalReason.INTENT_UNKNOWN
        assert _approval_rows(state) == [] and _approval_events(state) == 0


def test_a_review_of_a_denied_intent_holds(app) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.review_value import REVIEW_VALUE_ID

    with TestClient(app) as client:
        state = client.app.state
        _gid, _decision, intent, _forecast = _forward(state)
        for action in ("freeze", "unfreeze"):
            assert client.post(f"/api/values/{REVIEW_VALUE_ID}/{action}").status_code == 200
        assert state.policy_owner._submit_policy(intent.intent_id).result(timeout=30).value.verdict
        denied = state.approval_owner.review(intent.intent_id)
        assert denied.reason is R18ApprovalReason.POLICY_DENIED


# ============================================================ the confirm


def test_confirm_writes_one_challenge_and_a_retry_reuses_it(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        timers = _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        reviewed, issued = _confirm(state, intent.intent_id)
        assert (issued.status, issued.attempt) == (ApprovalStatus.ISSUED, 1)
        challenge = reviewed.challenge
        assert issued.challenge == challenge
        written = _adopted(state, ApprovalChallengeV1)
        assert written == [challenge]
        rows = _approval_rows(state)
        assert [(row.operation_kind, row.phase) for row in rows] == [
            (HistoryOperationKind.R18_CHALLENGE_ISSUE, HistoryPhase.FINALIZED)
        ]
        # The envelope carries the handler-computed attempt.
        from suzka.r16.history_records import HistoryRecordOperationV1

        assert rows[0].pending_bytes is not None
        assert HistoryRecordOperationV1.from_bytes(rows[0].pending_bytes).attempt == 1
        # One timer at expires + 1 s.
        assert [t.started for t in timers] == [True]
        assert abs(timers[0].delay - 301) <= 1
        # A review returns the WRITTEN challenge: same nonce, digest, code, attempt.
        again = owner.review(intent.intent_id)
        assert (again.status, again.challenge, again.attempt) == (ReviewStatus.WRITTEN, challenge, 1)
        # A confirm for the written live nonce writes nothing (-> the CLAIM, 17c).
        events = _approval_events(state)
        existing = owner.issue(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (existing.status, existing.attempt) == (ApprovalStatus.EXISTING, 1)
        assert _approval_events(state) == events and len(_approval_rows(state)) == 1
        wrong = owner.issue(intent.intent_id, challenge.nonce, "WRONG000")
        assert wrong.reason is R18ApprovalReason.CODE_MISMATCH
        # Inside the window the obligation is still pending: row 2.
        assert ip16b._preview(state, _gid).row is SuccessorRow.ROW_2
        _no_failed_events(state)


def test_a_wrong_code_or_unknown_nonce_writes_nothing_and_uses_no_attempt(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None
        nonce = review.challenge.nonce
        # The code check is in the event: the handler HOLDs, nothing written.
        bad = owner.issue(intent.intent_id, nonce, "AAAAAAAA")
        assert (bad.status, bad.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.CODE_MISMATCH)
        unknown = owner.issue(intent.intent_id, "e" * 64, review.challenge.confirmation_code)
        assert unknown.reason is R18ApprovalReason.UNKNOWN_CHALLENGE
        other = owner.issue("intent-" + "1" * 64, nonce, "AAAAAAAA")
        assert other.reason is R18ApprovalReason.INTENT_UNKNOWN
        # A challenge a caller built is not the pending one: the handler refuses it.
        forged = replace(review.challenge, nonce="f" * 64)
        held = owner._submit_issue(forged, forged.confirmation_code).result(timeout=30).value
        assert held.reason is R18ApprovalReason.UNKNOWN_CHALLENGE
        assert _approval_rows(state) == []
        # Declining is not confirming: the pending challenge simply lapses and
        # the next attempt is still number 1 (HD-4 (a)).
        assert owner.review(intent.intent_id).attempt == 1
        good = owner.issue(intent.intent_id, nonce, review.challenge.confirmation_code)
        assert (good.status, good.attempt) == (ApprovalStatus.ISSUED, 1)
        _no_failed_events(state)


# ================================================================ expiry


def test_expiry_then_a_new_attempt_then_three_close_the_intent(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        seen = ip16b._Seen(monkeypatch)
        gid, intent, _forecast = _allowed(state)
        for attempt in range(1, MAX_CHALLENGES_PER_INTENT + 1):
            reviewed, issued = _confirm(state, intent.intent_id)
            assert (issued.status, issued.attempt) == (ApprovalStatus.ISSUED, attempt)
            challenge = reviewed.challenge
            assert challenge is not None
            # Still live: the reason must be true.
            still = owner._submit_expire(challenge.challenge_digest).result(timeout=30).value
            assert (still.status, still.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.STILL_LIVE)
            # A second review is the written one, never a replacement.
            assert owner.review(intent.intent_id).challenge == challenge
            assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
            clock.advance(301)
            ended = _expire(state, challenge)
            assert ended.status is ApprovalStatus.EXPIRED
            assert ended.expiration.reason is ExpirationReason.EXPIRED
            assert _expire(state, challenge).status is ApprovalStatus.EXPIRE_EXISTING
            assert len(_adopted(state, type(ended.expiration))) == attempt
        # The 4th challenge is refused: three expiries closed the Intent.
        closed = owner.review(intent.intent_id)
        assert (closed.status, closed.reason) == (ReviewStatus.HOLD, R18ApprovalReason.LEDGER_UNSAFE)
        # §8a: 3 expiries -> row 2c, no Human action.
        _assert_row_2c(ip16b._preview(state, gid, seen))
        assert seen.last.ledger is LedgerState.APPROVAL_EXPIRED_UNCLAIMED
        _no_failed_events(state)


def test_a_dead_earlier_challenge_is_expired_by_the_next_confirm(app) -> None:  # type: ignore[no-untyped-def]
    """§3 step 5 (the lazy path): dead and unended -> expire first, then issue."""

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        first, _ = _confirm(state, intent.intent_id)
        clock.advance(400)  # dead, and no timer has run
        reviewed = owner.review(intent.intent_id)
        assert reviewed.status is ReviewStatus.BUILT and reviewed.attempt == 2
        assert reviewed.challenge is not None
        issued = owner.issue(intent.intent_id, reviewed.challenge.nonce, reviewed.challenge.confirmation_code)
        assert (issued.status, issued.attempt) == (ApprovalStatus.ISSUED, 2)
        assert [r.operation_kind for r in _approval_rows(state)] == [
            HistoryOperationKind.R18_CHALLENGE_ISSUE,
            HistoryOperationKind.R18_APPROVAL_EXPIRE,
            HistoryOperationKind.R18_CHALLENGE_ISSUE,
        ]
        assert [e.reason for e in _adopted(state, ApprovalExpirationV1)] == [
            ExpirationReason.EXPIRED
        ]
        assert first.challenge is not None
        _no_failed_events(state)


def test_a_live_written_challenge_is_never_replaced(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        first, _ = _confirm(state, intent.intent_id)
        assert first.challenge is not None
        # A second challenge built for the same Intent cannot be written (the
        # handler wants the pending one, and a live one is open).
        other = replace(first.challenge, nonce="a" * 64)
        held = owner._submit_issue(other, other.confirmation_code).result(timeout=30).value
        assert held.status is ApprovalStatus.HOLD
        with owner._lock:
            owner._pending_challenges[intent.intent_id] = other
        held = owner._submit_issue(other, other.confirmation_code).result(timeout=30).value
        assert held.reason is R18ApprovalReason.CHALLENGE_ALREADY_OPEN
        assert len(_approval_rows(state)) == 1


def test_the_timer_expires_the_challenge_a_second_after_its_end(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        timers = _fake_timers(state)
        clock = _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        _confirm(state, intent.intent_id)
        assert len(timers) == 1
        # Early (the clock is behind): STILL_LIVE is retried with a new timer.
        timers[0].callback()
        assert _adopted(state, ApprovalExpirationV1) == []
        assert len(timers) == 2 and timers[1].started
        clock.advance(301)
        timers[1].callback()
        written = _adopted(state, ApprovalExpirationV1)
        assert [e.reason for e in written] == [ExpirationReason.EXPIRED]
        assert len(timers) == 2  # done: no further timer
        # stop() cancels what is armed.
        state.approval_owner._arm(_adopted(state, ApprovalChallengeV1)[0], 0)
        armed = timers[-1]
        state.approval_owner.stop()
        assert armed.cancelled


def test_a_real_timer_fires_after_its_delay(app) -> None:  # type: ignore[no-untyped-def]
    """The default timer is a daemon ``threading.Timer`` (one real second)."""

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _gid, intent, _forecast = _allowed(state)
        reviewed, issued = _confirm(state, intent.intent_id)
        assert issued.status is ApprovalStatus.ISSUED and reviewed.challenge is not None
        assert owner._timers  # armed by the real factory
        owner.stop()
        assert owner._timers == {}


def test_a_restart_voids_the_open_challenge_and_uses_no_attempt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """HD-10 (b): the boot id is per API process; the startup scan records
    BOOT_CHANGED; the restart alone used no attempt (§8a)."""

    settings = _settings(tmp_path / "restart")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _fake_timers(state)
        gid, intent, _forecast = _allowed(state)
        reviewed, issued = _confirm(state, intent.intent_id)
        assert issued.status is ApprovalStatus.ISSUED and reviewed.challenge is not None
        challenge = reviewed.challenge
        boot = state.approval_owner.boot_id
        assert challenge.boot_id == boot
    with TestClient(_app(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        assert owner.boot_id != boot and owner.boot_id != challenge.boot_id
        ip16a2._until(lambda: len(_adopted(state, ApprovalExpirationV1)) == 1, timeout=30)
        (ended,) = _adopted(state, ApprovalExpirationV1)
        assert (ended.challenge_digest, ended.reason) == (
            challenge.challenge_digest,
            ExpirationReason.BOOT_CHANGED,
        )
        # The old challenge is dead for this boot, whatever the clock says.
        assert not challenge.is_live(datetime.now(timezone.utc).strftime(UTC_FORMAT), owner.boot_id)
        dead = owner.issue(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert dead.reason is R18ApprovalReason.CHALLENGE_NOT_LIVE
        # A new review is attempt 2 (1 written + 1); the first used 1 attempt.
        again = owner.review(intent.intent_id)
        assert (again.status, again.attempt) == (ReviewStatus.BUILT, 2)
        assert again.challenge is not None and again.challenge.boot_id == owner.boot_id
        _no_failed_events(state)


def test_restarts_without_a_confirm_use_no_attempt_and_close_at_the_window(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "restarts")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        gid, intent, _forecast = _allowed(state)
        assert state.approval_owner.review(intent.intent_id).challenge is not None
    for _ in range(2):
        with TestClient(_app(settings)) as client:
            state = client.app.state
            assert _adopted(state, ApprovalChallengeV1) == []
            assert state.approval_owner.review(intent.intent_id).attempt == 1
    with TestClient(_app(settings)) as client:
        state = client.app.state
        ip17a._assert_row_2c(ip17a._closed_preview(state, gid, intent, monkeypatch))


def test_the_startup_scan_skips_live_ended_and_claimed_challenges(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, forecast = _allowed(state)
        verdict = [r for r in ip17a._adopted(state) if r.record_kind is RecordKind.POLICY_VERDICT][0].record
        # Fixture challenges of another boot: attempt 1 claimed, attempt 2 open.
        one = ip16b._challenge(state, intent, verdict, 1, forecast)
        ip16b._claim(state, one, "2026-11-16T00:05:00Z")
        scan = owner._submit_expire_scan(None).result(timeout=30).value
        assert scan.status is ApprovalStatus.NONE_LEFT
        consumed = owner._submit_expire(one.challenge_digest).result(timeout=30).value
        assert consumed.reason is R18ApprovalReason.ALREADY_CONSUMED
        unknown = owner._submit_expire("0" * 64).result(timeout=30).value
        assert unknown.reason is R18ApprovalReason.CHALLENGE_UNKNOWN
        stray = owner._submit_expire_scan("0" * 64).result(timeout=30).value
        assert stray.reason is R18ApprovalReason.CHALLENGE_UNKNOWN and stray.challenge_digest is None


def test_the_scan_chains_over_every_dead_challenge(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _gid, intent, forecast = _allowed(state)
        verdict = [r for r in ip17a._adopted(state) if r.record_kind is RecordKind.POLICY_VERDICT][0].record
        one = ip16b._challenge(state, intent, verdict, 1, forecast)
        # Another boot -> dead: the scan ends it (BOOT_CHANGED), then none left.
        first = owner._submit_expire_scan(None).result(timeout=30).value
        assert (first.status, first.challenge_digest) == (ApprovalStatus.EXPIRED, one.challenge_digest)
        assert first.expiration.reason is ExpirationReason.BOOT_CHANGED
        two = ip16b._challenge(state, intent, verdict, 2, forecast)
        second = owner._submit_expire_scan(one.challenge_digest).result(timeout=30).value
        assert (second.status, second.challenge_digest) == (ApprovalStatus.EXPIRED, two.challenge_digest)
        assert owner._submit_expire_scan(two.challenge_digest).result(timeout=30).value.status is (
            ApprovalStatus.NONE_LEFT
        )


# ========================================================== §3b pre-checks


def test_every_insert_condition_holds_before_prepare(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        store = state.subject_history_store
        _gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None
        challenge, code = review.challenge, review.challenge.confirmation_code

        def hold(reason: R18ApprovalReason) -> None:
            result = owner._submit_issue(challenge, code).result(timeout=30).value
            assert (result.status, result.reason) == (ApprovalStatus.HOLD, reason)
            assert _approval_rows(state) == []
            assert owner._pending is None

        # §3b: a reserved but unadopted key (UNIQUE(intent, attempt) / nonce).
        real_keys = store.live_keys
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "live_keys",
                lambda keys: frozenset(k for k in keys if k.startswith("challenge:")) | real_keys(keys),
            )
            hold(R18ApprovalReason.IN_FLIGHT)
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "live_keys",
                lambda keys: frozenset(k for k in keys if k.startswith("nonce:")) | real_keys(keys),
            )
            hold(R18ApprovalReason.IN_FLIGHT)

        def full(length):  # type: ignore[no-untyped-def]
            raise HistoryStoreCapacityError("full")

        with monkeypatch.context() as patch:
            patch.setattr(store, "check_headroom", full)
            hold(R18ApprovalReason.HISTORY_CAPACITY)
        # §2-7: the AdmissionGate.
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_gate", SimpleNamespace(is_open=lambda: False))
            hold(R18ApprovalReason.ADMISSION_GATE_CLOSED)
            owner.invalidate(intent.intent_id)
            assert owner.review(intent.intent_id).reason is R18ApprovalReason.ADMISSION_GATE_CLOSED
        owner.invalidate(intent.intent_id)
        # The target cannot be opened / sources drift: HOLD, never a DENY.
        path = state.settings.r18.target_path
        moved = path.with_name("moved.sqlite3")
        path.rename(moved)
        try:
            hold(R18ApprovalReason.TARGET_UNAVAILABLE)
        finally:
            moved.rename(path)
        # An operation row of this Intent that is not adopted yet: IN_FLIGHT.
        with monkeypatch.context() as patch:
            real = SubjectHistoryReader._records

            def without(self, *args):  # type: ignore[no-untyped-def]
                return tuple(
                    item for item in real(self, *args) if item.record_kind is not RecordKind.POLICY_VERDICT
                )

            patch.setattr(SubjectHistoryReader, "_records", without)
            hold(R18ApprovalReason.POLICY_PENDING)
        # The challenge is stale when a bound value is mutated (N5).
        for field, value in (
            ("intent_digest", "1" * 64),
            ("operation_key", "2" * 64),
            ("payload_digest", "3" * 64),
            ("expectation_digest", "4" * 64),
            ("target_expected_revision", 9),
        ):
            mutated = replace(challenge, **{field: value})
            with owner._lock:
                owner._pending_challenges[intent.intent_id] = mutated
            result = owner._submit_issue(mutated, mutated.confirmation_code).result(timeout=30).value
            assert (result.status, result.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.CHALLENGE_STALE)
        deadline = replace(challenge, commitment_deadline_utc=(
            datetime.now(timezone.utc) + timedelta(days=2)
        ).strftime(UTC_FORMAT))
        with owner._lock:
            owner._pending_challenges[intent.intent_id] = deadline
        result = owner._submit_issue(deadline, deadline.confirmation_code).result(timeout=30).value
        assert result.reason is R18ApprovalReason.CHALLENGE_STALE
        # The window and the TTL.
        longer = replace(
            challenge,
            expires_utc=(datetime.fromisoformat(challenge.issued_utc.replace("Z", "+00:00")) + timedelta(seconds=800)).strftime(UTC_FORMAT),
        )
        with owner._lock:
            owner._pending_challenges[intent.intent_id] = longer
        assert owner._submit_issue(longer, longer.confirmation_code).result(timeout=30).value.reason in (
            R18ApprovalReason.CHALLENGE_STALE,
        )
        with owner._lock:
            owner._pending_challenges[intent.intent_id] = challenge
        # Untouched, it still works.
        good = owner._submit_issue(challenge, code).result(timeout=30).value
        assert good.status is ApprovalStatus.ISSUED
        again = owner._submit_issue(challenge, code).result(timeout=30).value
        assert again.reason in (R18ApprovalReason.NONCE_USED, R18ApprovalReason.CHALLENGE_ALREADY_OPEN)
        _no_failed_events(state)


def test_the_expiry_insert_conditions_hold_before_prepare(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _fake_timers(state)
        clock = _Clock(state)
        store = state.subject_history_store
        _gid, intent, _forecast = _allowed(state)
        reviewed, _issued = _confirm(state, intent.intent_id)
        challenge = reviewed.challenge
        assert challenge is not None
        clock.advance(301)
        real_keys = store.live_keys
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "live_keys",
                lambda keys: frozenset(k for k in keys if k.startswith("challenge-end:")) | real_keys(keys),
            )
            held = _expire(state, challenge)
            assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.IN_FLIGHT)
        assert len(_approval_rows(state, HistoryOperationKind.R18_APPROVAL_EXPIRE)) == 0
        assert _expire(state, challenge).status is ApprovalStatus.EXPIRED


# ======================================================= gating and bands


def test_disabled_r18_holds_before_any_read(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", r18=False))) as client:
        state = client.app.state
        owner = state.approval_owner
        assert type(owner) is ApprovalChallengeOwnerV1 and not owner.enabled
        assert owner._scan_thread is None

        def no_read(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("read while r18 is disabled")

        challenge = _challenge()
        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "classify", no_read)
            patch.setattr(state._plan_preview_owner, "_state_snapshot", no_read)
            results = [
                owner._submit_issue(challenge, "AAAAAAAA").result(timeout=30).value,
                owner._submit_expire("0" * 64).result(timeout=30).value,
                owner._submit_expire_scan(None).result(timeout=30).value,
                owner.issue("intent-" + "0" * 64, "0" * 64, "AAAAAAAA"),
            ]
            for held in results:
                assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.R18_DISABLED)
            assert owner.review("intent-" + "0" * 64).reason is R18ApprovalReason.R18_DISABLED
        assert _approval_rows(state) == []
        assert state.subject_history_store.has_record_operations() is False
        _no_failed_events(state)


def test_production_with_r18_off_posts_no_approval_event_or_thread(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "prod", r18=False))) as client:
        state = client.app.state
        assert _approval_events(state) == 0
        assert state.approval_owner._scan_thread is None and state.approval_owner._timers == {}
        assert state.r16_driver._policy is None
    # R18 on but an empty store: still no event (nothing can be unended).
    with TestClient(_app(_settings(tmp_path / "empty"))) as client:
        state = client.app.state
        assert _approval_events(state) == 0 and state.approval_owner._scan_thread is None


def test_no_owner_without_subject_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "nohistory")
    settings = settings.model_copy(
        update={"subject_history": settings.subject_history.model_copy(update={"enabled": False})}
    )
    with TestClient(_app(settings)) as client:
        assert client.app.state.approval_owner is None


def test_inverse_challenges_are_deferred(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        held = owner.issue("inverse-intent-" + "0" * 64, "0" * 64, "AAAAAAAA")
        assert held.reason is R18ApprovalReason.INVERSE_DEFERRED
        assert _approval_rows(state) == []


def test_the_soft_band_refuses_the_challenge_and_admits_the_expiry() -> None:
    assert AgentEventType.R18_CHALLENGE_ISSUE not in SOFT_BAND_ADMITTED_EVENT_TYPES
    assert AgentEventType.R18_APPROVAL_EXPIRE in SOFT_BAND_ADMITTED_EVENT_TYPES
    soft = EventJournalAdmissionStatus(
        True, "journal_soft_limit", 0, 1, 0, 1, True, False, False,
        band=EventJournalAdmissionBand.SOFT,
    )
    assert not soft.admits(AgentEventType.R18_CHALLENGE_ISSUE)
    assert soft.admits(AgentEventType.R18_APPROVAL_EXPIRE)
    assert soft.admits(AgentEventType.PLAN_PREVIEW_REVIEW)


def test_only_the_approval_owner_submits_the_approval_events() -> None:
    """No other module names the approval event types / source, or calls the
    owner's private submit methods; server.py has only its exact wiring."""

    import re

    root = Path(__file__).resolve().parents[1] / "suzka"
    types = re.compile(r"AgentEventType\.R18_(CHALLENGE_ISSUE|APPROVAL_EXPIRE)")
    source = re.compile(r"AgentEventSource\.(API_R18_APPROVAL|RUNTIME_R18_APPROVAL_EXPIRE)")
    submits = re.compile(r"\b_submit_(issue|expire|expire_scan)\b")
    # file -> the exact count of allowed type mentions (an event list or table).
    allowed_types = {"approval_owner.py": None, "event_journal.py": 1, "server.py": 2}
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name != "approval_owner.py":
            assert not source.search(text), path
            assert not submits.search(text), f"{path} calls a private approval submit"
        found = types.findall(text)
        if path.name in allowed_types:
            if allowed_types[path.name] is not None:
                assert len(found) == allowed_types[path.name], path
        else:
            assert not found, f"{path} uses an approval event type"
    server = (root / "api" / "server.py").read_text(encoding="utf-8")
    wiring = [line.strip() for line in server.splitlines() if types.search(line)]
    assert wiring == [
        "AgentEventType.R18_CHALLENGE_ISSUE,",
        "AgentEventType.R18_APPROVAL_EXPIRE,",
    ]
    assert "approval_owner.review" not in server and ".issue(" not in server


# ============================================== §8a: no stop hinders the subject


def test_drift_after_the_allow_holds_the_confirm_and_closes_at_the_window(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.review_value import REVIEW_VALUE_ID

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None
        for action in ("freeze", "unfreeze"):
            assert client.post(f"/api/values/{REVIEW_VALUE_ID}/{action}").status_code == 200
        held = owner.issue(intent.intent_id, review.challenge.nonce, review.challenge.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.SOURCE_NOT_CURRENT)
        assert _approval_rows(state) == []
        events = _approval_events(state)
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
        _assert_row_2c(ip17a._closed_preview(state, gid, intent, monkeypatch))
        assert _approval_events(state) == events  # no event closed it
        # After closure (a successor Decision) a confirm is refused for good.
        late = owner.review(intent.intent_id)
        assert late.status is ReviewStatus.HOLD
        _no_failed_events(state)


def test_a_confirm_after_a_successor_decision_is_refused_even_with_the_clock_set_back(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None
        _assert_row_2c(ip17a._closed_preview(state, gid, intent, monkeypatch))
        assert intent.window_open(owner._clock())  # the clock is back inside
        held = owner.issue(intent.intent_id, review.challenge.nonce, review.challenge.confirmation_code)
        assert held.status is ApprovalStatus.HOLD
        assert held.reason in (R18ApprovalReason.SOURCE_NOT_CURRENT, R18ApprovalReason.LEDGER_UNSAFE)
        assert _approval_rows(state) == []
        assert owner.review(intent.intent_id).status is ReviewStatus.HOLD
        assert DecisionIssueStatus.ISSUED is not None
        _no_failed_events(state)


def test_never_reviewed_and_never_confirmed_close_at_the_window(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        gid, intent, _forecast = _allowed(state)
        events = _approval_events(state)
        _assert_row_2c(ip17a._closed_preview(state, gid, intent, monkeypatch))
        assert _approval_events(state) == events and _approval_rows(state) == []
        _no_failed_events(state)


def test_a_crash_after_prepare_is_tombstoned_and_the_confirm_can_retry(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime.agent_runtime import AgentRuntimeDurabilityError

    settings = _settings(tmp_path / "crash")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R18_CHALLENGE_ISSUE:
                raise RuntimeError("simulated pre-WAL failure")

        owner._validate_commit = crash  # type: ignore[method-assign]
        # The runtime fail-stops; issue() itself never raises (a HOLD).
        held = owner.issue(
            intent.intent_id, review.challenge.nonce, review.challenge.confirmation_code
        )
        assert held.reason is R18ApprovalReason.JOURNAL_UNAVAILABLE
        # The runtime stopped: a further submit is refused, never a crash.
        from suzka.runtime.agent_runtime import AgentRuntimeStopped

        with pytest.raises(AgentRuntimeStopped):
            owner._submit_issue(review.challenge, review.challenge.confirmation_code)
        assert AgentRuntimeDurabilityError
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [r.phase for r in _approval_rows(state)] == [HistoryPhase.ABORTED]
        # Nothing was written: attempt 1 again, in a new process.
        assert state.approval_owner.review(intent.intent_id).attempt == 1


def test_validate_commit_sees_exactly_its_own_prepared_row(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        seen: list[str] = []
        real = owner._validate_commit

        def spy(event):  # type: ignore[no-untyped-def]
            if owner._pending is not None:
                seen.extend(row.phase.value for row in _approval_rows(state))
            return real(event)

        owner._validate_commit = spy  # type: ignore[method-assign]
        try:
            _reviewed, issued = _confirm(state, intent.intent_id)
        finally:
            owner._validate_commit = real  # type: ignore[method-assign]
        assert issued.status is ApprovalStatus.ISSUED
        assert seen == [HistoryPhase.PREPARED.value]
        assert owner._pending is None


def test_the_boot_id_is_a_fresh_uuid_per_owner(app) -> None:  # type: ignore[no-untyped-def]
    from uuid import UUID

    with TestClient(app) as client:
        owner = client.app.state.approval_owner
        assert str(UUID(owner.boot_id)) == owner.boot_id
        other = ApprovalChallengeOwnerV1(
            client.app.state.policy_owner,
            client.app.state.agent_runtime,
            enabled=False,
            ttl_seconds=60,
        )
        assert other.boot_id != owner.boot_id
        for bad in (0, 901, True, "300"):
            with pytest.raises(TypeError):
                ApprovalChallengeOwnerV1(
                    client.app.state.policy_owner,
                    client.app.state.agent_runtime,
                    enabled=False,
                    ttl_seconds=bad,  # type: ignore[arg-type]
                )


# ============================================ review fixes (independent review)


def _count_inspects(state, monkeypatch) -> list[int]:  # type: ignore[no-untyped-def]
    calls = [0]
    real = state.event_journal.inspect

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(state.event_journal, "inspect", counting)
    return calls


def test_rapid_reviews_are_coalesced_and_cannot_stall_the_journal(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Condition B: a caller looping on review does not loop the journal read
    (``inspect`` holds the journal lock)."""

    import threading

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        now = [1000.0]
        owner._monotonic = lambda: now[0]
        _gid, intent, _forecast = _allowed(state)
        calls = _count_inspects(state, monkeypatch)
        # A live (built, then pending) challenge: one read in total.
        for _ in range(100):
            assert owner.review(intent.intent_id).challenge is not None
        assert calls[0] <= 2
        # A HOLD (unknown Intent): one read per interval, however many callers.
        calls[0] = 0
        unknown = "intent-" + "0" * 64
        for _ in range(100):
            assert owner.review(unknown).reason is R18ApprovalReason.INTENT_UNKNOWN
        assert calls[0] <= 2
        now[0] += 2.5
        owner.review(unknown)
        assert calls[0] <= 4
        # Concurrent callers: one read at a time, the rest get the cached
        # result or REVIEW_BUSY.
        calls[0] = 0
        now[0] += 10
        results: list[Any] = []

        def loop() -> None:
            for _ in range(25):
                results.append(owner.review("intent-" + "1" * 64))

        threads = [threading.Thread(target=loop) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 200 and calls[0] <= 3
        assert {r.reason for r in results} <= {
            R18ApprovalReason.INTENT_UNKNOWN,
            R18ApprovalReason.REVIEW_BUSY,
        }
        assert _approval_rows(state) == [] and _approval_events(state) == 0


def test_a_live_written_challenge_is_shown_without_a_read(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        reviewed, issued = _confirm(state, intent.intent_id)
        calls = _count_inspects(state, monkeypatch)
        for _ in range(20):
            again = owner.review(intent.intent_id)
            assert (again.status, again.challenge, again.attempt) == (
                ReviewStatus.WRITTEN,
                reviewed.challenge,
                1,
            )
        assert calls[0] == 0 and issued.status is ApprovalStatus.ISSUED


def test_a_wrong_code_submits_no_event(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None
        events = _approval_events(state)
        for _ in range(5):
            held = owner.issue(intent.intent_id, review.challenge.nonce, "WRONG000")
            assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.CODE_MISMATCH)
        assert _approval_events(state) == events and _approval_rows(state) == []
        _reviewed, issued = _confirm(state, intent.intent_id)
        assert issued.status is ApprovalStatus.ISSUED
        events = _approval_events(state)
        held = owner.issue(intent.intent_id, review.challenge.nonce, "WRONG000")
        assert held.reason is R18ApprovalReason.CODE_MISMATCH
        assert _approval_events(state) == events


def test_issue_never_raises_on_a_refused_band_or_a_timeout(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from concurrent.futures import Future, TimeoutError as FutureTimeout

    from suzka.runtime.agent_runtime import AgentRuntimeStopped
    from suzka.runtime.event_journal import EventJournalError

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None
        args = (intent.intent_id, review.challenge.nonce, review.challenge.confirmation_code)

        def refused(challenge, code):  # type: ignore[no-untyped-def]
            future: Future[Any] = Future()
            future.set_exception(EventJournalError("journal_soft_limit"))
            return future

        def stopped(challenge, code):  # type: ignore[no-untyped-def]
            raise AgentRuntimeStopped(None)  # type: ignore[arg-type]

        def timeout(challenge, code):  # type: ignore[no-untyped-def]
            return Future()  # never completes

        for submit in (refused, stopped):
            monkeypatch.setattr(owner, "_submit_issue", submit)
            held = owner.issue(*args)
            assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.JOURNAL_UNAVAILABLE)
        monkeypatch.setattr(owner, "_submit_issue", timeout)

        def instant_timeout(future):  # type: ignore[no-untyped-def]
            raise FutureTimeout()

        monkeypatch.setattr(owner, "_run", instant_timeout)
        held = owner.issue(*args)
        assert held.reason is R18ApprovalReason.JOURNAL_UNAVAILABLE
        assert _approval_rows(state) == []
        # The pending challenge survives; the real path still works.
        monkeypatch.undo()
        _fake_timers(state)
        assert owner.issue(*args).status is ApprovalStatus.ISSUED


def test_start_survives_an_unreadable_store_and_the_scan_retries_once(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from concurrent.futures import Future

    from suzka.r18 import approval_owner as module
    from suzka.runtime.agent_runtime import AgentEventOutcome

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        owner.stop()
        owner._scan_thread = None

        def broken() -> bool:
            raise RuntimeError("store closed")

        monkeypatch.setattr(state.subject_history_store, "has_record_operations", broken)
        owner.start()
        assert owner._scan_thread is None
        monkeypatch.undo()
        # The scan: one retry of a transient HOLD that names no challenge.
        monkeypatch.setattr(module, "STARTUP_SCAN_RETRY_DELAY", 0.01)
        script = [
            owner._hold(R18ApprovalReason.HISTORY_UNAVAILABLE, None),
            ApprovalResultFactory.none_left(),
        ]
        seen: list[Any] = []

        def fake_scan(after):  # type: ignore[no-untyped-def]
            seen.append(after)
            future: Future[Any] = Future()
            future.set_result(AgentEventOutcome(ip16a2._event(), script.pop(0)))
            return future

        monkeypatch.setattr(owner, "_submit_expire_scan", fake_scan)
        owner._stopped = False
        owner._startup_scan()
        assert seen == [None, None] and script == []
        # A second transient HOLD ends the scan (documented gap: the lazy path
        # and the window cover it).
        script[:] = [
            owner._hold(R18ApprovalReason.HISTORY_UNAVAILABLE, None),
            owner._hold(R18ApprovalReason.HISTORY_UNAVAILABLE, None),
            ApprovalResultFactory.none_left(),
        ]
        seen.clear()
        owner._startup_scan()
        assert seen == [None, None] and len(script) == 1
        # A non-transient HOLD without a digest is not retried.
        script[:] = [owner._hold(R18ApprovalReason.R18_DISABLED, None)]
        seen.clear()
        owner._startup_scan()
        assert seen == [None]


class ApprovalResultFactory:
    @staticmethod
    def none_left():  # type: ignore[no-untyped-def]
        from suzka.r18.approval_owner import ApprovalResult

        return ApprovalResult(ApprovalStatus.NONE_LEFT, None)


def test_a_lost_nonce_race_returns_the_written_challenge(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        reviewed, issued = _confirm(state, intent.intent_id)
        challenge = reviewed.challenge
        assert challenge is not None and issued.status is ApprovalStatus.ISSUED
        # The racing confirm still believes the nonce is only pending.
        real = owner._existing
        first = [True]

        def lagging(intent_id, nonce, code):  # type: ignore[no-untyped-def]
            if first[0]:
                first[0] = False
                return None
            return real(intent_id, nonce, code)

        monkeypatch.setattr(owner, "_existing", lagging)
        owner._shown.clear()
        events = _approval_events(state)
        result = owner.issue(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (result.status, result.challenge, result.attempt) == (
            ApprovalStatus.EXISTING,
            challenge,
            1,
        )
        assert _approval_events(state) == events + 1 and len(_approval_rows(state)) == 1


def test_a_confirm_holds_on_a_target_mismatch_and_on_exhausted_attempts(app) -> None:  # type: ignore[no-untyped-def]

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        # Three written challenges, two ended, the third dead and unended: a
        # fourth attempt is refused (the Intent is still open: 2 expiries).
        for attempt in (1, 2):
            reviewed, issued = _confirm(state, intent.intent_id)
            assert issued.attempt == attempt and reviewed.challenge is not None
            clock.advance(301)
            assert _expire(state, reviewed.challenge).status is ApprovalStatus.EXPIRED
        reviewed, issued = _confirm(state, intent.intent_id)
        assert issued.attempt == 3
        clock.advance(301)
        exhausted = owner.review(intent.intent_id)
        assert (exhausted.status, exhausted.reason) == (
            ReviewStatus.HOLD,
            R18ApprovalReason.ATTEMPTS_EXHAUSTED,
        )
        assert len(_approval_rows(state, HistoryOperationKind.R18_CHALLENGE_ISSUE)) == 3

    # The same for a target that changed after the review.
    # (A fresh app: another store.)


def test_a_confirm_holds_on_a_target_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from suzka.r18.target import TargetStore

    with TestClient(_app(_settings(tmp_path / "mismatch"))) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        review = owner.review(intent.intent_id)
        assert review.challenge is not None
        target = TargetStore.open(state.settings.r18.target_path)
        try:
            target.fence(intent.obligation_group_key, Direction.FORWARD)
        finally:
            target.close()
        owner.invalidate(intent.intent_id)
        held = owner.issue(intent.intent_id, review.challenge.nonce, review.challenge.confirmation_code)
        assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.TARGET_MISMATCH)
        assert _approval_rows(state) == []
        # A review no longer shows it (cache dropped by the HOLD; interval ok).
        owner._monotonic = lambda: 10**9
        assert owner.review(intent.intent_id).reason is R18ApprovalReason.TARGET_MISMATCH


def test_an_unadopted_prepared_row_is_in_flight(app) -> None:  # type: ignore[no-untyped-def]
    """The real path: a PREPARED operation row of this Intent that is not
    adopted holds its keys and its attempt count."""

    from uuid import uuid4

    from suzka.r16.history_records import (
        ApprovalExpirationV1 as Expiration,
        HistoryRecordOperationV1,
    )
    from suzka.r16.history_store import HistoryStoreError
    from suzka.runtime.event_journal import TransactionKind

    with TestClient(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        clock = _Clock(state)
        store = state.subject_history_store
        _gid, intent, _forecast = _allowed(state)
        reviewed, _issued = _confirm(state, intent.intent_id)
        challenge = reviewed.challenge
        assert challenge is not None
        clock.advance(301)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R18_APPROVAL_EXPIRE,
            ip10._incarnation(state),
            str(uuid4()),
            10**6,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.APPROVAL_EXPIRATION,
            Expiration.for_challenge(challenge, ExpirationReason.EXPIRED),
        )
        store.insert_prepared_record(
            transaction_id=str(uuid4()),
            operation_digest=operation.operation_digest,
            operation=operation,
        )
        assert [r.phase for r in _approval_rows(state, HistoryOperationKind.R18_APPROVAL_EXPIRE)] == [
            HistoryPhase.PREPARED
        ]
        kinds = frozenset({HistoryOperationKind.R18_APPROVAL_EXPIRE})
        try:
            classified = owner._policy._classify()
        except Exception as blocked:  # a stray PREPARED row blocks the reader
            from suzka.r18.policy_owner import _PolicyHold

            assert isinstance(blocked, _PolicyHold)
            # The reader refuses to classify at all (a global block), so the
            # handler HOLDs earlier than ``_in_flight`` can: not reachable.
            held = _expire(state, challenge)
            assert (held.status, held.reason) in (
                (ApprovalStatus.HOLD, R18ApprovalReason.HISTORY_UNAVAILABLE),
                (ApprovalStatus.HOLD, R18ApprovalReason.HISTORY_CONFLICT),
            )
        else:
            assert owner._in_flight(classified, intent.intent_id, kinds) is True
            held = _expire(state, challenge)
            assert held.status is ApprovalStatus.HOLD
        assert HistoryStoreError  # imported for the store contract
        # Why the real path is not reachable through the server: the reader
        # blocks the whole incarnation on a PREPARED row that has no journal
        # transaction, so the handler HOLDs at the classification, before
        # ``_in_flight``.  ``_in_flight`` (a FINALIZED-but-not-adopted row, a
        # race) is therefore tested on the row the store really holds.
        row = _approval_rows(state, HistoryOperationKind.R18_APPROVAL_EXPIRE)[0]
        fake = SimpleNamespace(records=(), rows=(row,))
        assert owner._in_flight(fake, intent.intent_id, kinds) is True  # type: ignore[arg-type]
        assert owner._in_flight(fake, "intent-" + "9" * 64, kinds) is False  # type: ignore[arg-type]
        adopted = SimpleNamespace(
            records=(SimpleNamespace(record=Expiration.for_challenge(challenge, ExpirationReason.EXPIRED)),),
            rows=(row,),
        )
        assert owner._in_flight(adopted, intent.intent_id, kinds) is False  # type: ignore[arg-type]
