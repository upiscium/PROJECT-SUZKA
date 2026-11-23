"""IP-16a-2: the R18 ActionIntent issuer, its event and the driver's Intent
stage on the real server (#411 comment 6916372789 §1-§3, §7)."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from datetime import timedelta, timezone
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16 import decision_issuer as decision_module
from suzka.r16.decision_issuer import DecisionIssueStatus, _DecisionIssueResult
from suzka.r16.driver import R16Driver
from suzka.r16.goal_owner import GoalOrigin as R16GoalOrigin
from suzka.r16.history_codec import PolicyRow
from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import HistoryOperationKind, RecordKind
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError
from suzka.r16.successor_policy import SuccessorRow
from suzka.r18.contracts import (
    ActionIntentV1,
    Direction,
    GateStatus,
    GoalOrigin,
    SourceRole,
)
from suzka.r18.intent_issuer import (
    IntentIssueStatus,
    R18IntentReason,
    ReviewArtifactIntentIssuerV1,
    _IntentIssueResult,
)
from suzka.r18.obligation import SECRET_FILE, load_obligation_secret, obligation_identity
from suzka.r18.target import TargetReader, TargetStore
from suzka.r18.target_init import initialize
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
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
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_commitment, _qualified_goal

X_TEXT = "source experience; fixture"  # the text _qualified_goal sends as X


def _target_path(root: Path) -> Path:
    directory = root / "target"
    if not directory.exists():
        directory.mkdir(mode=0o700, parents=True)
    os.chmod(directory, 0o700)
    return directory / "target.sqlite3"


def _settings(root: Path, *, r18: bool = True, init: bool = True):  # type: ignore[no-untyped-def]
    base = ip10._settings(root)
    path = _target_path(root)
    if init and not path.exists():
        initialize(path)
    return base.model_copy(
        update={"r18": base.r18.model_copy(update={"enabled": r18, "target_path": path})}
    )


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip16a2"))


def _issue_decision(state, goal_id):  # type: ignore[no-untyped-def]
    return ip10._issue(state, goal_id)


def _row4(state):  # type: ignore[no-untyped-def]
    x, gid = _qualified_goal(state)
    first = _issue_decision(state, gid)
    assert first.row is SuccessorRow.ROW_3
    _qualified_commitment(state, x)
    second = _issue_decision(state, gid)
    assert (second.status, second.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_4)
    assert second.lineage_scope == second.decision.lineage_scope
    return x, gid, second


def _second_goal(state) -> str:  # type: ignore[no-untyped-def]
    """Another adopted Goal on a new X with the same text."""

    from suzka.r16.goal_owner import _GoalSourceSelection
    from suzka.r16.outcomes import R16OwnerStatus
    from test_r16_u2b_goal_owner import chat, x_id

    x = chat(state, X_TEXT)
    assert state._motivation_source_owner._submit(x).result(timeout=30).value == "interest_applied"
    p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
    owner = state._goal_source_owner
    proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(timeout=30).value
    adopted = owner._submit_review(proposed.subject_id).result(timeout=30).value
    assert adopted.status is R16OwnerStatus.ADOPTED
    state._ip16_second_x = x
    return proposed.subject_id


def _intent(state, scope):  # type: ignore[no-untyped-def]
    return state.intent_issuer._submit_intent(scope).result(timeout=30).value


def _intent_rows(state):  # type: ignore[no-untyped-def]
    return [
        row
        for row in state.subject_history_store.operations(ip10._incarnation(state))
        if row.operation_kind is HistoryOperationKind.R18_INTENT_ISSUE
    ]


def _expected_group(state) -> str:  # type: ignore[no-untyped-def]
    path = state.settings.r18.target_path
    return obligation_identity(
        load_obligation_secret(path.parent),
        subject_id=state.subject_history_store.meta().subject_id,
        context_id="conversation.default",
        request_text=X_TEXT,
        scope=("record_review_explanation", "record_source_qualification"),
    ).group_key


def _no_failed_events(state) -> None:  # type: ignore[no-untyped-def]
    records = state.event_journal.inspect().records
    assert not any(record.lifecycle is EventLifecycle.FAILED for record in records)


# ------------------------------------------------------------- the issuer


def test_row_4_issues_one_adopted_intent_then_existing(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        assert type(state.intent_issuer) is ReviewArtifactIntentIssuerV1
        assert state.r16_driver._intent is state.intent_issuer
        _x, _gid, decision = _row4(state)
        before = len(ip10._history_transactions(state))
        issued = _intent(state, decision.lineage_scope)
        assert type(issued) is _IntentIssueResult
        assert issued.status is IntentIssueStatus.ISSUED
        intent = issued.intent
        assert type(intent) is ActionIntentV1 and issued.intent_id == intent.intent_id
        committed = decision.decision
        assert (intent.decision_id, intent.decision_committed_digest) == (
            committed.decision_id,
            committed.content_digest,
        )
        assert intent.policy_row is PolicyRow.ROW_4
        assert intent.policy_kind is committed.policy_kind
        assert intent.selected_candidate_id == "candidate-b"
        assert intent.selected_candidate_digest == committed.decision.selected.candidate_digest
        assert intent.goal_origin is GoalOrigin.USER_REQUEST
        assert intent.target_expected_revision == 0
        reader = TargetReader.open(state.settings.r18.target_path)
        try:
            assert intent.target_instance_id == reader.meta().target_instance_id
        finally:
            reader.close()
        artifact = intent.artifact
        assert [gate.status for gate in artifact.gates] == [GateStatus.PASS] * 6
        assert [item.role for item in artifact.sources] == sorted(SourceRole)
        assert artifact.source(SourceRole.DECISION).revision == 2  # head depth
        assert intent.obligation_group_key == _expected_group(state)
        # One more R07 transaction, adopted and readable through the reader.
        assert len(ip10._history_transactions(state)) == before + 1
        adopted = SubjectHistoryReader(
            state.event_journal, state.subject_history_store
        ).adopted_records()
        assert adopted.blocked is None
        assert [(r.record_kind, r.record) for r in adopted.records] == [
            (RecordKind.ACTION_INTENT, intent)
        ]
        assert state.intent_issuer._pending is None
        # The request text is in neither stored file (only inside the HMAC).
        for path in (
            state.settings.subject_history.path,
            state.settings.r18.target_path,
        ):
            assert X_TEXT.encode() not in path.read_bytes()
        # A resubmit is EXISTING, with no second transaction.
        again = _intent(state, decision.lineage_scope)
        assert (again.status, again.intent_id) == (IntentIssueStatus.EXISTING, intent.intent_id)
        assert len(ip10._history_transactions(state)) == before + 1
        # IP-16b: the PLAN_PREVIEW afterwards reads the ledger (INTENT_ISSUED)
        # and holds at row 2; no Decision is issued.
        after = _issue_decision(state, _gid)
        assert (after.status, after.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2)
        _no_failed_events(state)


def test_row_3_row_4b_and_an_empty_lineage_give_no_intent(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        first = _issue_decision(state, gid)
        assert first.row is SuccessorRow.ROW_3
        held = _intent(state, first.lineage_scope)
        assert (held.status, held.reason) == (IntentIssueStatus.HOLD, R18IntentReason.NOT_ROW_4)
        assert _intent(state, "x-" + "0" * 64).reason is R18IntentReason.NOT_ROW_4
        _qualified_commitment(state, x)
        monkeypatch.setattr(
            state.decision_issuer,
            "_interpretation",
            lambda *args: SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTESTED"),
        )
        row4b = _issue_decision(state, gid)
        assert row4b.row is SuccessorRow.ROW_4B
        assert _intent(state, row4b.lineage_scope).reason is R18IntentReason.NOT_ROW_4
        assert _intent_rows(state) == []
        with pytest.raises(ValueError):
            state.intent_issuer._submit_intent("not-a-scope")
        _no_failed_events(state)


def test_an_unverifiable_head_is_a_head_conflict(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        monkeypatch.setattr(state.decision_issuer, "_head_verified", lambda *args: False)
        held = _intent(state, decision.lineage_scope)
        assert held.reason is R18IntentReason.HEAD_CONFLICT
        assert _intent_rows(state) == []


def test_self_origin_goal_never_yields_an_intent(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        preview = state._plan_preview_owner
        real = preview._qualify
        monkeypatch.setattr(
            preview,
            "_qualify",
            lambda goal_id, event: replace(
                real(goal_id, event), goal_origin=R16GoalOrigin.SUBJECT_MOTIVATION
            ),
        )
        held = _intent(state, decision.lineage_scope)
        assert held.reason is R18IntentReason.SELF_ORIGIN_GOAL
        assert _intent_rows(state) == []


def test_commitment_deadline_margin_holds(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        _issue_decision(state, gid)
        commitment_id = _qualified_commitment(state, x)
        decision = _issue_decision(state, gid)
        at = state.main_loop._commitment_system.get(commitment_id).deadline.at
        if at is None:
            pytest.skip("fixture Commitment has no deadline")
        issuer = state.intent_issuer
        monkeypatch.setattr(issuer, "_clock", lambda: at - decision_module.DEADLINE_MARGIN / 2)
        assert _intent(state, decision.lineage_scope).reason is R18IntentReason.DEADLINE_ELAPSED
        monkeypatch.setattr(issuer, "_clock", lambda: at + decision_module.DEADLINE_MARGIN)
        assert _intent(state, decision.lineage_scope).reason is R18IntentReason.DEADLINE_ELAPSED
        assert _intent_rows(state) == []
        # IP-17a §3a: a window shorter than MIN_APPROVAL_WINDOW (it ends at
        # ``deadline - DEADLINE_MARGIN``) is DEADLINE_ELAPSED too.
        from suzka.r18.intent_issuer import MIN_APPROVAL_WINDOW

        short = at - decision_module.DEADLINE_MARGIN - MIN_APPROVAL_WINDOW + timedelta(seconds=1)
        monkeypatch.setattr(issuer, "_clock", lambda: short)
        assert _intent(state, decision.lineage_scope).reason is R18IntentReason.DEADLINE_ELAPSED
        assert _intent_rows(state) == []
        enough = at - decision_module.DEADLINE_MARGIN - MIN_APPROVAL_WINDOW - timedelta(seconds=1)
        monkeypatch.setattr(issuer, "_clock", lambda: enough)
        issued = _intent(state, decision.lineage_scope)
        assert issued.status is IntentIssueStatus.ISSUED
        # The window ends at the deadline margin (floored to the second).
        end = (at - decision_module.DEADLINE_MARGIN).astimezone(timezone.utc).replace(microsecond=0)
        assert issued.intent.approval_window_end_utc == end.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_drifted_sources_hold(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        real = decision_module._context_digest
        monkeypatch.setattr(decision_module, "_context_digest", lambda q: "e" * 64 if real(q) else real(q))
        held = _intent(state, decision.lineage_scope)
        assert held.reason is R18IntentReason.SOURCE_DRIFT
        assert _intent_rows(state) == []


def test_deadline_is_checked_again_after_the_slow_reads(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Review P2: the clock enters the margin while the target is opened."""

    from suzka.r18 import intent_issuer as intent_module

    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        _issue_decision(state, gid)
        commitment_id = _qualified_commitment(state, x)
        decision = _issue_decision(state, gid)
        at = state.main_loop._commitment_system.get(commitment_id).deadline.at
        if at is None:
            pytest.skip("fixture Commitment has no deadline")
        # IP-17a: early enough for a window of at least MIN_APPROVAL_WINDOW.
        now = [at - decision_module.DEADLINE_MARGIN - timedelta(minutes=10)]
        issuer = state.intent_issuer
        monkeypatch.setattr(issuer, "_clock", lambda: now[0])
        real_open = intent_module.TargetReader.open

        def slow_open(path):  # type: ignore[no-untyped-def]
            now[0] = at - decision_module.DEADLINE_MARGIN / 2
            return real_open(path)

        monkeypatch.setattr(intent_module.TargetReader, "open", staticmethod(slow_open))
        held = _intent(state, decision.lineage_scope)
        assert (held.status, held.reason) == (
            IntentIssueStatus.HOLD,
            R18IntentReason.DEADLINE_ELAPSED,
        )
        assert _intent_rows(state) == []
        assert issuer._pending is None
        _no_failed_events(state)


def test_a_real_value_revision_move_is_source_drift(app) -> None:  # type: ignore[no-untyped-def]
    """Review P3: no patched digest; the review Value really moves (freeze
    then unfreeze) after the row-4 Decision, so its sealed ref no longer
    matches the requalified source."""

    from suzka.r16.review_value import REVIEW_VALUE_ID

    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        before = [
            ref for ref in decision.decision.source_refs if ref.kind == "value"
        ][0].revision
        for action in ("freeze", "unfreeze"):
            response = client.post(f"/api/values/{REVIEW_VALUE_ID}/{action}")
            assert response.status_code == 200, response.text
        assert response.json()["revision"] != before
        held = _intent(state, decision.lineage_scope)
        assert (held.status, held.reason) == (IntentIssueStatus.HOLD, R18IntentReason.SOURCE_DRIFT)
        assert _intent_rows(state) == []
        _no_failed_events(state)


def test_a_suspended_and_resumed_context_is_unavailable(app) -> None:  # type: ignore[no-untyped-def]
    """A real Context transition after the Decision: requalification through
    the preview owner fails, so the real path is SOURCE_UNAVAILABLE (the
    drift comparison is never reached)."""

    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        context_id = state.main_loop.context_registry.state.current_context_id
        for action in ("suspend", "resume"):
            response = client.post(f"/api/contexts/{context_id}/{action}")
            assert response.status_code == 200, response.text
        held = _intent(state, decision.lineage_scope)
        assert (held.status, held.reason) == (
            IntentIssueStatus.HOLD,
            R18IntentReason.SOURCE_UNAVAILABLE,
        )
        assert _intent_rows(state) == []


def test_missing_target_and_secret_hold_and_nothing_is_created(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        path = state.settings.r18.target_path
        secret = path.parent / SECRET_FILE
        os.chmod(secret, 0o644)
        assert _intent(state, decision.lineage_scope).reason is (
            R18IntentReason.OBLIGATION_SECRET_UNAVAILABLE
        )
        os.chmod(secret, 0o600)
        data = secret.read_bytes()
        secret.unlink()
        assert _intent(state, decision.lineage_scope).reason is (
            R18IntentReason.OBLIGATION_SECRET_UNAVAILABLE
        )
        assert not secret.exists()
        fd = os.open(secret, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, data)
        os.close(fd)
        moved = path.with_name("moved.sqlite3")
        path.rename(moved)
        assert _intent(state, decision.lineage_scope).reason is R18IntentReason.TARGET_UNAVAILABLE
        assert not path.exists()  # never created
        assert _intent_rows(state) == []
        moved.rename(path)
        assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.ISSUED


def test_a_target_row_for_the_group_is_a_prior_obligation(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        target = TargetStore.open(state.settings.r18.target_path)
        try:
            target.fence(_expected_group(state), Direction.FORWARD)
        finally:
            target.close()
        held = _intent(state, decision.lineage_scope)
        assert held.reason is R18IntentReason.PRIOR_OBLIGATION
        assert _intent_rows(state) == []


def test_capacity_holds_before_prepare(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)

        def full(length):  # type: ignore[no-untyped-def]
            raise HistoryStoreCapacityError("full")

        with monkeypatch.context() as patch:
            patch.setattr(state.subject_history_store, "check_headroom", full)
            assert _intent(state, decision.lineage_scope).reason is R18IntentReason.HISTORY_CAPACITY
        assert _intent_rows(state) == []
        assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.ISSUED


def test_a_reserved_but_unadopted_key_is_in_flight(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        store = state.subject_history_store
        real = store.live_keys
        monkeypatch.setattr(
            store,
            "live_keys",
            lambda keys: frozenset(key for key in keys if key.startswith("operation:")) | real(keys),
        )
        assert _intent(state, decision.lineage_scope).reason is R18IntentReason.IN_FLIGHT
        monkeypatch.setattr(
            store,
            "live_keys",
            lambda keys: frozenset(key for key in keys if key.startswith("claim:")),
        )
        assert _intent(state, decision.lineage_scope).reason is R18IntentReason.PRIOR_OBLIGATION
        assert _intent_rows(state) == []


def test_identical_request_is_one_obligation_across_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§8-1 (a): the same text in the same Context is the same group key; a
    second lineage for it cannot issue a second Intent."""

    settings = _settings(tmp_path / "same")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        first = _intent(state, decision.lineage_scope)
        assert first.status is IntentIssueStatus.ISSUED
    with TestClient(_app(settings)) as client:
        state = client.app.state
        again = _intent(state, decision.lineage_scope)
        assert (again.status, again.intent_id) == (IntentIssueStatus.EXISTING, first.intent_id)
        assert again.intent.obligation_group_key == first.intent.obligation_group_key
        assert _expected_group(state) == first.intent.obligation_group_key
        # A new X with the same text: a new lineage, the same obligation.
        # IP-16b: the Decision issuer's ledger reader holds it at row 0 ...
        gid = _second_goal(state)
        first_preview = _issue_decision(state, gid)
        assert (first_preview.status, first_preview.row) == (
            DecisionIssueStatus.HOLD,
            SuccessorRow.ROW_0,
        )
        # ... and behind it the Intent issuer's own guard still refuses a
        # second Intent (a pre-IP-16b ledger witness reaches row 4 here).
        from suzka.r18.ledger import NO_LEDGER

        issuer = state.decision_issuer
        issuer._ledger = lambda *args: NO_LEDGER  # type: ignore[method-assign]
        try:
            assert _issue_decision(state, gid).row is SuccessorRow.ROW_3
            _qualified_commitment(state, state._ip16_second_x)
            other = _issue_decision(state, gid)
            assert other.row is SuccessorRow.ROW_4
        finally:
            del issuer._ledger
        assert other.lineage_scope != decision.lineage_scope
        held = _intent(state, other.lineage_scope)
        assert held.reason is R18IntentReason.PRIOR_OBLIGATION
        assert len(_intent_rows(state)) == 1


def test_validate_commit_sees_exactly_its_own_prepared_row(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        issuer = state.intent_issuer
        seen: list[str] = []
        real = issuer._validate_commit

        def spy(event):  # type: ignore[no-untyped-def]
            if issuer._pending is not None:
                seen.extend(row.phase.value for row in _intent_rows(state))
            return real(event)

        issuer._validate_commit = spy  # type: ignore[method-assign]
        try:
            assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.ISSUED
        finally:
            issuer._validate_commit = real  # type: ignore[method-assign]
        assert seen == [HistoryPhase.PREPARED.value]


def test_crash_after_prepare_is_tombstoned_then_reissued(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "crash-prepare")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R18_ACTION_INTENT_ISSUE:
                raise RuntimeError("simulated pre-WAL failure")

        state.intent_issuer._validate_commit = crash  # type: ignore[method-assign]
        with pytest.raises(AgentRuntimeDurabilityError):
            _intent(state, decision.lineage_scope)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _intent_rows(state)] == [HistoryPhase.ABORTED]
        issued = _intent(state, decision.lineage_scope)
        assert issued.status is IntentIssueStatus.ISSUED


def test_crash_after_internal_commit_rolls_forward_then_existing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.history_participant import SubjectHistoryRecordParticipant

    settings = _settings(tmp_path / "crash-finalize")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        real = SubjectHistoryRecordParticipant.finalize

        def crash(self, binding):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated post-R06 failure")

        SubjectHistoryRecordParticipant.finalize = crash  # type: ignore[method-assign]
        try:
            with pytest.raises(AgentRuntimeDurabilityError):
                _intent(state, decision.lineage_scope)
        finally:
            SubjectHistoryRecordParticipant.finalize = real  # type: ignore[method-assign]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _intent_rows(state)] == [HistoryPhase.FINALIZED]
        assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.EXISTING


def test_disabled_r18_holds_every_event_and_the_driver_gets_no_issuer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", r18=False))) as client:
        state = client.app.state
        issuer = state.intent_issuer
        assert type(issuer) is ReviewArtifactIntentIssuerV1 and not issuer.enabled
        assert state.r16_driver._intent is None
        _x, _gid, decision = _row4(state)
        held = _intent(state, decision.lineage_scope)
        assert (held.status, held.reason) == (IntentIssueStatus.HOLD, R18IntentReason.R18_DISABLED)
        assert _intent_rows(state) == []
        assert state.subject_history_store.has_record_operations() is False
        _no_failed_events(state)


def test_no_issuer_without_subject_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "nohistory")
    settings = settings.model_copy(
        update={"subject_history": settings.subject_history.model_copy(update={"enabled": False})}
    )
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.intent_issuer is None
        assert state.r16_driver._intent is None


def test_the_soft_band_refuses_the_intent_event() -> None:
    assert AgentEventType.R18_ACTION_INTENT_ISSUE not in SOFT_BAND_ADMITTED_EVENT_TYPES
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
    assert not soft.admits(AgentEventType.R18_ACTION_INTENT_ISSUE)


# ------------------------------------------------------------- the driver


class _FakeRuntime:
    def pending_count(self) -> int:
        return 0


class _FakeOwner:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.calls: list[tuple[str, object]] = []
        self.futures: list[Future[Any]] = []

    def _record(self, name: str, argument: object) -> Future[Any]:
        self.calls.append((name, argument))
        future: Future[Any] = Future()
        self.futures.append(future)
        return future

    def _submit_preview(self, goal_id: str) -> Future[Any]:
        return self._record("preview", goal_id)

    def _submit_intent(self, scope: str) -> Future[Any]:
        return self._record("intent", scope)


def _event() -> AgentEvent:
    import datetime
    import uuid

    return AgentEvent(
        str(uuid.uuid4()),
        AgentEventType.PLAN_PREVIEW_REVIEW,
        AgentEventSource.RUNTIME_PLAN_PREVIEW_REVIEW,
        datetime.datetime.now(datetime.UTC),
        7,
    )


def _decision_result(row: SuccessorRow, status=DecisionIssueStatus.ISSUED, decision_id="d1"):  # type: ignore[no-untyped-def]
    fake = SimpleNamespace(decision_id=decision_id, lineage_scope="x-" + "a" * 64)
    return _DecisionIssueResult(status, decision_id, row, None, None, fake)  # type: ignore[arg-type]


def _until(predicate, timeout: float = 10.0):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _driver(intent: _FakeOwner | None):  # type: ignore[no-untyped-def]
    """A driver whose thread is not started: steps are run synchronously."""

    preview = _FakeOwner()
    owners = [_FakeOwner() for _ in range(3)]
    clock = _Clock()
    driver = R16Driver(
        _FakeRuntime(),  # type: ignore[arg-type]
        *owners,  # type: ignore[arg-type]
        preview,  # type: ignore[arg-type]
        active=True,
        clock=clock,
        intent_issuer=intent,  # type: ignore[arg-type]
    )
    return driver, preview, clock


def _drain(driver: R16Driver) -> None:
    finished, driver._done = driver._done, []
    for item in finished:
        driver._on_done(item)
    driver._service_waiting()


def _previewed(driver: R16Driver, preview: _FakeOwner, value: object) -> None:
    driver._maybe_preview(f"goal-{len(preview.calls)}", "commitment_accepted")
    preview.futures[-1].set_result(AgentEventOutcome(_event(), value))
    _drain(driver)


def test_driver_submits_one_intent_per_row_4_decision() -> None:
    intent = _FakeOwner()
    driver, preview, clock = _driver(intent)
    _previewed(driver, preview, _decision_result(SuccessorRow.ROW_4))
    assert intent.calls == [("intent", "x-" + "a" * 64)]
    # The same Decision again (EXISTING) is not resubmitted.
    _previewed(driver, preview, _decision_result(SuccessorRow.ROW_4, DecisionIssueStatus.EXISTING))
    # Other rows never are.
    _previewed(driver, preview, _decision_result(SuccessorRow.ROW_3, decision_id="d2"))
    _previewed(driver, preview, _decision_result(SuccessorRow.ROW_4B, decision_id="d3"))
    assert len(intent.calls) == 1
    # An in_flight HOLD is retried with the bounded delays, at most 3 times.
    for attempt in range(4):
        intent.futures[-1].set_result(
            AgentEventOutcome(
                _event(),
                _IntentIssueResult(IntentIssueStatus.HOLD, None, R18IntentReason.IN_FLIGHT),
            )
        )
        _drain(driver)
        clock.now += 10
        _drain(driver)
        assert len(intent.calls) == min(attempt + 2, 4)
    # Any other HOLD only ends the stage.
    assert len(intent.calls) == 4


def test_driver_does_not_retry_other_holds() -> None:
    intent = _FakeOwner()
    driver, preview, clock = _driver(intent)
    _previewed(driver, preview, _decision_result(SuccessorRow.ROW_4))
    intent.futures[-1].set_result(
        AgentEventOutcome(
            _event(),
            _IntentIssueResult(IntentIssueStatus.HOLD, None, R18IntentReason.TARGET_UNAVAILABLE),
        )
    )
    _drain(driver)
    clock.now += 10
    _drain(driver)
    assert len(intent.calls) == 1


@pytest.mark.parametrize("intent", (None, _FakeOwner(enabled=False)))
def test_driver_without_an_enabled_issuer_submits_nothing(intent) -> None:  # type: ignore[no-untyped-def]
    driver, preview, _clock = _driver(intent)
    _previewed(driver, preview, _decision_result(SuccessorRow.ROW_4))
    assert intent is None or intent.calls == []
    assert driver.counters["submitted"] == 1  # the preview only


def _chat(client: TestClient, text: str) -> dict[str, Any]:
    response = client.post("/api/chat", json={"message": text, "attachments": []})
    assert response.status_code == 200, response.text
    return response.json()


def _intent_events(state) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type is AgentEventType.R18_ACTION_INTENT_ISSUE
        and row.lifecycle is EventLifecycle.COMPLETED
    )


@pytest.mark.parametrize("enabled", (True, False))
def test_driver_chain_ends_in_one_intent_only_when_enabled(tmp_path, enabled) -> None:  # type: ignore[no-untyped-def]
    from test_r16_ip6b_driver import _interest, _settled

    with TestClient(_app(_settings(tmp_path / "chain", r18=enabled))) as client:
        state = client.app.state
        assert client.post("/api/r16/driver/activate").status_code == 200
        x = _chat(client, "something wonderful happened today")
        _interest(client, x["experience_id"])
        review = _chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}")
        assert _settled(client, review["command_ticket"]["ticket_id"])["status"] == "adopted"
        commit = _chat(client, f"COMMIT_REVIEW_EXPERIENCE:1:{x['experience_id']}")
        assert _settled(client, commit["command_ticket"]["ticket_id"])["status"] == "accepted"

        def previews() -> int:
            return sum(
                1
                for row in state.event_journal.records
                if row.event_type is AgentEventType.PLAN_PREVIEW_REVIEW
                and row.lifecycle is EventLifecycle.COMPLETED
            )

        _until(lambda: previews() >= 2, timeout=30)
        if enabled:
            _until(lambda: _intent_events(state) == 1, timeout=30)
        time.sleep(0.5)
        assert _intent_events(state) == (1 if enabled else 0)
        assert len(_intent_rows(state)) == (1 if enabled else 0)
