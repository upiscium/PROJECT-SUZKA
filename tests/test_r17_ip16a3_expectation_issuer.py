"""IP-16a-3: the R17 expectation (forecast) issuer, its event and the
driver's forecast stage on the real server (#411 D1 r3 §1, §2, §4, §7)."""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import HistoryOperationKind, RecordKind
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError
from suzka.r16.successor_policy import SuccessorRow
from suzka.r17 import expectation_issuer as expectation_module
from suzka.r17.contracts import ExpectedObservationV1
from suzka.r17.expectation_issuer import (
    MAX_SCANS_PER_STARTUP,
    ExpectationCommitStatus,
    ExpectationIssuerV1,
    R17ExpectationReason,
    _ExpectationResult,
)
from suzka.r18.contracts import payload_digest
from suzka.r18.intent_issuer import IntentIssueStatus, R18IntentReason, _IntentIssueResult
from suzka.runtime.agent_runtime import (
    AgentEventOutcome,
    AgentEventType,
    AgentRuntimeDurabilityError,
)
from suzka.runtime.event_journal import (
    SOFT_BAND_ADMITTED_EVENT_TYPES,
    EventJournalAdmissionBand,
    EventJournalAdmissionStatus,
    EventLifecycle,
)

import test_r16_ip10_decision_issuer as ip10
import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_u2a_source_owner import _app

ADMIN = ip16a2.ADMIN
_settings = ip16a2._settings
_row4 = ip16a2._row4
_intent = ip16a2._intent
_intent_rows = ip16a2._intent_rows
_no_failed_events = ip16a2._no_failed_events


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip16a3"))


@pytest.fixture
def _admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", ip16a2.TOKEN)


def _forecast(state, intent_id):  # type: ignore[no-untyped-def]
    return state.expectation_issuer._submit_expectation(intent_id).result(timeout=30).value


def _scan(state):  # type: ignore[no-untyped-def]
    return _forecast(state, None)


def _forecast_rows(state):  # type: ignore[no-untyped-def]
    return [
        row
        for row in state.subject_history_store.operations(ip10._incarnation(state))
        if row.operation_kind is HistoryOperationKind.R17_EXPECTATION_COMMIT
    ]


def _adopted(state):  # type: ignore[no-untyped-def]
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return found.records


def _issued_intent(state):  # type: ignore[no-untyped-def]
    _x, gid, decision = _row4(state)
    issued = _intent(state, decision.lineage_scope)
    assert issued.status is IntentIssueStatus.ISSUED
    return gid, decision, issued.intent


def _forecast_events(state) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type is AgentEventType.R17_EXPECTATION_COMMIT
        and row.lifecycle is EventLifecycle.COMPLETED
    )


# ------------------------------------------------------------- the issuer


def test_terminal_intent_gets_exactly_one_bound_forecast(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        assert type(state.expectation_issuer) is ExpectationIssuerV1
        assert state.r16_driver._expectation is state.expectation_issuer
        _gid, decision, intent = _issued_intent(state)
        before = len(ip10._history_transactions(state))
        committed = _forecast(state, intent.intent_id)
        assert type(committed) is _ExpectationResult
        assert committed.status is ExpectationCommitStatus.COMMITTED
        forecast = committed.forecast
        assert type(forecast) is ExpectedObservationV1
        assert (committed.intent_id, committed.forecast_id) == (
            intent.intent_id,
            forecast.forecast_id,
        )
        # Bound to the terminal Decision and the Intent (values read here).
        sealed = decision.decision
        forecast.check_binding(
            decision_id=sealed.decision_id,
            decision_committed_digest=sealed.content_digest,
            policy_kind=sealed.policy_kind,
            policy_row=sealed.policy_row,
            action_intent_id=intent.intent_id,
            action_intent_digest=intent.intent_digest,
            forward_key=intent.forward_key,
            target_instance_id=intent.target_instance_id,
            target_namespace=intent.target_namespace,
            target_expected_revision=intent.target_expected_revision,
        )
        assert forecast.predicted_payload_digest == intent.payload_digest
        assert forecast.predicted_payload_digest == payload_digest(
            intent.artifact.payload_bytes()
        )
        assert (forecast.lineage_scope, forecast.incarnation_id) == (
            intent.lineage_scope,
            intent.incarnation_id,
        )
        assert forecast.obligation_group_key == intent.obligation_group_key
        assert (forecast.selected_candidate_id, forecast.selected_candidate_digest) == (
            sealed.decision.selected.candidate_id,
            sealed.decision.selected.candidate_digest,
        )
        assert intent.admitting_sequence <= forecast.as_of_r07_sequence
        assert forecast.as_of_r07_sequence < forecast.admitting_sequence
        # One more R07 transaction; the forecast is adopted.
        assert len(ip10._history_transactions(state)) == before + 1
        assert [row.phase for row in _forecast_rows(state)] == [HistoryPhase.FINALIZED]
        assert [(r.record_kind, r.record) for r in _adopted(state)] == [
            (RecordKind.ACTION_INTENT, intent),
            (RecordKind.ACTION_FORECAST, forecast),
        ]
        assert state.expectation_issuer._pending is None
        # Explicit resubmit and scan are both idempotent: no new transaction.
        again = _forecast(state, intent.intent_id)
        assert (again.status, again.forecast_id) == (
            ExpectationCommitStatus.EXISTING,
            forecast.forecast_id,
        )
        scan = _scan(state)
        assert (scan.status, scan.intent_id, scan.reason) == (
            ExpectationCommitStatus.NONE_LEFT,
            None,
            None,
        )
        assert len(ip10._history_transactions(state)) == before + 1
        assert len(_forecast_rows(state)) == 1
        _no_failed_events(state)


def test_scan_commits_a_missing_forecast_then_none_left(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        assert _scan(state).status is ExpectationCommitStatus.NONE_LEFT
        _gid, _decision, intent = _issued_intent(state)
        scanned = _scan(state)
        assert (scanned.status, scanned.intent_id) == (
            ExpectationCommitStatus.COMMITTED,
            intent.intent_id,
        )
        assert _scan(state).status is ExpectationCommitStatus.NONE_LEFT
        explicit = _forecast(state, intent.intent_id)
        assert (explicit.status, explicit.forecast_id) == (
            ExpectationCommitStatus.EXISTING,
            scanned.forecast_id,
        )
        assert len(_forecast_rows(state)) == 1


def test_unknown_and_not_yet_terminal_intents_hold(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        unknown = _forecast(state, "intent-" + "0" * 64)
        assert (unknown.status, unknown.reason) == (
            ExpectationCommitStatus.HOLD,
            R17ExpectationReason.INTENT_UNKNOWN,
        )
        _gid, _decision, intent = _issued_intent(state)
        # The Intent's row exists (FINALIZED) but the reader does not adopt
        # it, as in the live window before its event terminal.
        real = SubjectHistoryReader._records

        def unadopted(self, *args):  # type: ignore[no-untyped-def]
            return tuple(
                record
                for record in real(self, *args)
                if record.record_kind is not RecordKind.ACTION_INTENT
            )

        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "_records", unadopted)
            held = _forecast(state, intent.intent_id)
            assert (held.status, held.reason, held.intent_id) == (
                ExpectationCommitStatus.HOLD,
                R17ExpectationReason.NOT_YET_TERMINAL,
                intent.intent_id,
            )
            # The scan sees only adopted Intents.
            assert _scan(state).status is ExpectationCommitStatus.NONE_LEFT
        assert _forecast_rows(state) == []
        assert _forecast(state, intent.intent_id).status is ExpectationCommitStatus.COMMITTED
        _no_failed_events(state)


def test_binding_mismatch_is_refused_without_a_write(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, _decision, intent = _issued_intent(state)
        real = ExpectationIssuerV1._decision_for

        def other_decision(classified, target):  # type: ignore[no-untyped-def]
            found = real(classified, target)
            return SimpleNamespace(
                decision_id=found.decision_id,
                content_digest="e" * 64,
                policy_kind=found.policy_kind,
                policy_row=found.policy_row,
                lineage_scope=found.lineage_scope,
                decision=found.decision,
            )

        with monkeypatch.context() as patch:
            patch.setattr(ExpectationIssuerV1, "_decision_for", staticmethod(other_decision))
            held = _forecast(state, intent.intent_id)
            assert (held.status, held.reason) == (
                ExpectationCommitStatus.HOLD,
                R17ExpectationReason.BINDING_MISMATCH,
            )
            assert _scan(state).reason is R17ExpectationReason.BINDING_MISMATCH
        with monkeypatch.context() as patch:
            patch.setattr(expectation_module, "_predicted_payload_digest", lambda intent: "f" * 64)
            held = _forecast(state, intent.intent_id)
            assert (held.status, held.reason) == (
                ExpectationCommitStatus.HOLD,
                R17ExpectationReason.PAYLOAD_MISMATCH,
            )
        assert _forecast_rows(state) == []
        assert state.expectation_issuer._pending is None
        assert _forecast(state, intent.intent_id).status is ExpectationCommitStatus.COMMITTED
        _no_failed_events(state)


def test_in_flight_keys_and_capacity_hold(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, _decision, intent = _issued_intent(state)
        store = state.subject_history_store
        real = store.live_keys
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "live_keys",
                lambda keys: frozenset(k for k in keys if k.startswith("forecast-key:")) | real(keys),
            )
            assert _forecast(state, intent.intent_id).reason is R17ExpectationReason.IN_FLIGHT

        def full(length):  # type: ignore[no-untyped-def]
            raise HistoryStoreCapacityError("full")

        with monkeypatch.context() as patch:
            patch.setattr(store, "check_headroom", full)
            assert _forecast(state, intent.intent_id).reason is (
                R17ExpectationReason.HISTORY_CAPACITY
            )
        assert _forecast_rows(state) == []
        assert _forecast(state, intent.intent_id).status is ExpectationCommitStatus.COMMITTED


def test_validate_commit_sees_exactly_its_own_prepared_row(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, _decision, intent = _issued_intent(state)
        issuer = state.expectation_issuer
        seen: list[str] = []
        real = issuer._validate_commit

        def spy(event):  # type: ignore[no-untyped-def]
            if issuer._pending is not None:
                seen.extend(row.phase.value for row in _forecast_rows(state))
            return real(event)

        issuer._validate_commit = spy  # type: ignore[method-assign]
        try:
            assert _forecast(state, intent.intent_id).status is ExpectationCommitStatus.COMMITTED
        finally:
            issuer._validate_commit = real  # type: ignore[method-assign]
        assert seen == [HistoryPhase.PREPARED.value]


def test_crash_after_prepare_is_tombstoned_then_recommitted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "crash-prepare")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _gid, _decision, intent = _issued_intent(state)

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R17_EXPECTATION_COMMIT:
                raise RuntimeError("simulated pre-WAL failure")

        state.expectation_issuer._validate_commit = crash  # type: ignore[method-assign]
        with pytest.raises(AgentRuntimeDurabilityError):
            _forecast(state, intent.intent_id)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _forecast_rows(state)] == [HistoryPhase.ABORTED]
        assert _forecast(state, intent.intent_id).status is ExpectationCommitStatus.COMMITTED


def test_crash_after_internal_commit_rolls_forward_then_existing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.history_participant import SubjectHistoryRecordParticipant

    settings = _settings(tmp_path / "crash-finalize")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _gid, _decision, intent = _issued_intent(state)
        real = SubjectHistoryRecordParticipant.finalize

        def crash(self, binding):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated post-R06 failure")

        SubjectHistoryRecordParticipant.finalize = crash  # type: ignore[method-assign]
        try:
            with pytest.raises(AgentRuntimeDurabilityError):
                _forecast(state, intent.intent_id)
        finally:
            SubjectHistoryRecordParticipant.finalize = real  # type: ignore[method-assign]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _forecast_rows(state)] == [HistoryPhase.FINALIZED]
        assert _forecast(state, intent.intent_id).status is ExpectationCommitStatus.EXISTING
        assert _scan(state).status is ExpectationCommitStatus.NONE_LEFT


def test_disabled_r18_holds_without_reading_and_the_driver_gets_no_issuer(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", r18=False))) as client:
        state = client.app.state
        issuer = state.expectation_issuer
        assert type(issuer) is ExpectationIssuerV1 and not issuer.enabled
        assert state.r16_driver._expectation is None
        _row4(state)

        def no_read(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("read while r18 is disabled")

        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "classify", no_read)
            patch.setattr(state.subject_history_store, "live_keys", no_read)
            for request in ("intent-" + "1" * 64, None):
                held = _forecast(state, request)
                assert (held.status, held.reason) == (
                    ExpectationCommitStatus.HOLD,
                    R17ExpectationReason.R18_DISABLED,
                )
        assert _forecast_rows(state) == []
        assert state.subject_history_store.has_record_operations() is False
        _no_failed_events(state)


def test_no_issuer_without_subject_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "nohistory")
    settings = settings.model_copy(
        update={"subject_history": settings.subject_history.model_copy(update={"enabled": False})}
    )
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.expectation_issuer is None
        assert state.r16_driver._expectation is None


def test_the_soft_band_refuses_the_forecast_event() -> None:
    assert AgentEventType.R17_EXPECTATION_COMMIT not in SOFT_BAND_ADMITTED_EVENT_TYPES
    soft = EventJournalAdmissionStatus(
        True,
        "journal_soft_limit",
        0,
        1,
        0,
        1,
        True,
        False,
        False,
        band=EventJournalAdmissionBand.SOFT,
    )
    assert soft.admits(AgentEventType.PLAN_PREVIEW_REVIEW)
    assert not soft.admits(AgentEventType.R17_EXPECTATION_COMMIT)


# ------------------------------------------------------------- restart scan


def test_crash_between_intent_and_forecast_is_completed_by_the_restart_scan(tmp_path, _admin_token) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "restart-scan")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        # The driver is activated, but its forecast stage is cut off, as if
        # the process ended right after the Intent's event.
        state.r16_driver._expectation = None
        _gid, _decision, intent = _issued_intent(state)
        assert client.post("/api/r16/driver/activate", headers=ADMIN).status_code == 200
        time.sleep(0.5)
        assert _forecast_rows(state) == []
        assert _forecast_events(state) == 0
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.r16_driver.active
        ip16a2._until(lambda: len(_forecast_rows(state)) == 1, timeout=30)
        ip16a2._until(lambda: _forecast_events(state) == 2, timeout=30)  # COMMITTED, NONE_LEFT
        time.sleep(0.5)
        assert _forecast_events(state) == 2
        adopted = _adopted(state)
        assert [r.record_kind for r in adopted] == [
            RecordKind.ACTION_INTENT,
            RecordKind.ACTION_FORECAST,
        ]
        assert adopted[1].record.action_intent_id == intent.intent_id
        _no_failed_events(state)


def test_driver_chain_ends_in_one_forecast_only_when_enabled(tmp_path, _admin_token) -> None:  # type: ignore[no-untyped-def]
    from test_r16_ip6b_driver import _interest, _settled

    for enabled in (True, False):
        with TestClient(_app(_settings(tmp_path / f"chain-{enabled}", r18=enabled))) as client:
            state = client.app.state
            assert client.post("/api/r16/driver/activate", headers=ADMIN).status_code == 200
            x = ip16a2._chat(client, "something wonderful happened today")
            _interest(client, x["experience_id"])
            review = ip16a2._chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}")
            assert _settled(client, review["command_ticket"]["ticket_id"])["status"] == "adopted"
            commit = ip16a2._chat(client, f"COMMIT_REVIEW_EXPERIENCE:1:{x['experience_id']}")
            assert _settled(client, commit["command_ticket"]["ticket_id"])["status"] == "accepted"
            if enabled:
                ip16a2._until(lambda: len(_forecast_rows(state)) == 1, timeout=30)
                # The activation scan and the explicit forecast both settle
                # (under load the scan may run after the Intent: then it
                # commits, scans again to NONE_LEFT and the explicit one
                # reads EXISTING).
                ip16a2._until(lambda: _forecast_events(state) >= 2, timeout=30)
            else:
                ip16a2._until(
                    lambda: sum(
                        1
                        for row in state.event_journal.records
                        if row.event_type is AgentEventType.PLAN_PREVIEW_REVIEW
                        and row.lifecycle is EventLifecycle.COMPLETED
                    )
                    >= 2,
                    timeout=30,
                )
            time.sleep(0.5)
            if enabled:
                assert _forecast_events(state) in (2, 3)
            else:
                assert _forecast_events(state) == 0
            assert len(_forecast_rows(state)) == (1 if enabled else 0)
            assert len(_intent_rows(state)) == (1 if enabled else 0)
            _no_failed_events(state)


# ------------------------------------------------------------- the driver


class _FakeExpectation(ip16a2._FakeOwner):
    def _submit_expectation(self, intent_id: str | None) -> Future[Any]:
        return self._record("expectation", intent_id)


def _driver():  # type: ignore[no-untyped-def]
    from suzka.r16.driver import R16Driver

    intent = ip16a2._FakeOwner()
    expectation = _FakeExpectation()
    preview = ip16a2._FakeOwner()
    clock = ip16a2._Clock()
    driver = R16Driver(
        ip16a2._FakeRuntime(),  # type: ignore[arg-type]
        *[ip16a2._FakeOwner() for _ in range(3)],  # type: ignore[arg-type]
        preview,  # type: ignore[arg-type]
        active=True,
        clock=clock,
        intent_issuer=intent,  # type: ignore[arg-type]
        expectation_issuer=expectation,  # type: ignore[arg-type]
    )
    return driver, preview, intent, expectation, clock


def _resolve(future: Future[Any], value: object) -> None:
    future.set_result(AgentEventOutcome(ip16a2._event(), value))


def _forecast_result(status, intent_id="intent-a", reason=None):  # type: ignore[no-untyped-def]
    if status in (ExpectationCommitStatus.COMMITTED, ExpectationCommitStatus.EXISTING):
        fake = SimpleNamespace(forecast_id="forecast-a", action_intent_id=intent_id)
        return _ExpectationResult(status, intent_id, "forecast-a", None, fake)  # type: ignore[arg-type]
    return _ExpectationResult(status, intent_id, None, reason)


@pytest.mark.parametrize("status", (IntentIssueStatus.ISSUED, IntentIssueStatus.EXISTING))
def test_driver_submits_a_forecast_after_any_intent_result(status) -> None:  # type: ignore[no-untyped-def]
    driver, preview, intent, expectation, clock = _driver()
    ip16a2._previewed(driver, preview, ip16a2._decision_result(SuccessorRow.ROW_4))
    fake = SimpleNamespace(intent_id="intent-a")
    _resolve(intent.futures[-1], _IntentIssueResult(status, "intent-a", None, fake))  # type: ignore[arg-type]
    ip16a2._drain(driver)
    assert expectation.calls == [("expectation", "intent-a")]
    # A not-yet-terminal HOLD is retried with the bounded delays (3 times).
    for attempt in range(4):
        _resolve(
            expectation.futures[-1],
            _forecast_result(
                ExpectationCommitStatus.HOLD,
                reason=R17ExpectationReason.NOT_YET_TERMINAL,
            ),
        )
        ip16a2._drain(driver)
        clock.now += 10
        ip16a2._drain(driver)
        assert len(expectation.calls) == min(attempt + 2, 4)
    assert all(call == ("expectation", "intent-a") for call in expectation.calls)


def test_driver_does_not_forecast_after_an_intent_hold_or_retry_other_holds() -> None:
    driver, preview, intent, expectation, clock = _driver()
    ip16a2._previewed(driver, preview, ip16a2._decision_result(SuccessorRow.ROW_4))
    _resolve(
        intent.futures[-1],
        _IntentIssueResult(IntentIssueStatus.HOLD, None, R18IntentReason.TARGET_UNAVAILABLE),
    )
    ip16a2._drain(driver)
    assert expectation.calls == []
    driver._maybe_forecast("intent-b")
    _resolve(
        expectation.futures[-1],
        _forecast_result(
            ExpectationCommitStatus.HOLD,
            "intent-b",
            R17ExpectationReason.BINDING_MISMATCH,
        ),
    )
    ip16a2._drain(driver)
    clock.now += 10
    ip16a2._drain(driver)
    assert expectation.calls == [("expectation", "intent-b")]
    driver._maybe_forecast("intent-b")  # once per Intent this incarnation
    assert len(expectation.calls) == 1


def test_driver_scan_is_bounded_and_stops_at_none_left_or_hold() -> None:
    driver, _preview, _intent, expectation, _clock = _driver()
    driver._maybe_scan()
    while expectation.futures and not expectation.futures[-1].done():
        _resolve(expectation.futures[-1], _forecast_result(ExpectationCommitStatus.COMMITTED))
        ip16a2._drain(driver)
    assert len(expectation.calls) == MAX_SCANS_PER_STARTUP == 16
    assert all(call == ("expectation", None) for call in expectation.calls)
    for final in (
        _forecast_result(ExpectationCommitStatus.NONE_LEFT, None),
        _forecast_result(
            ExpectationCommitStatus.HOLD, None, R17ExpectationReason.HISTORY_UNAVAILABLE
        ),
    ):
        driver, _preview, _intent, expectation, _clock = _driver()
        driver._maybe_scan()
        _resolve(expectation.futures[-1], _forecast_result(ExpectationCommitStatus.COMMITTED))
        ip16a2._drain(driver)
        _resolve(expectation.futures[-1], final)
        ip16a2._drain(driver)
        assert len(expectation.calls) == 2


def test_driver_posts_the_scan_once_and_only_when_active_and_enabled() -> None:
    driver, _preview, _intent, _expectation, _clock = _driver()
    driver._post_forecast_scan()
    driver._post_forecast_scan()
    assert driver._queue.qsize() == 1
    disabled, _preview, _intent, expectation, _clock = _driver()
    expectation.enabled = False
    disabled._post_forecast_scan()
    assert disabled._queue.qsize() == 0
