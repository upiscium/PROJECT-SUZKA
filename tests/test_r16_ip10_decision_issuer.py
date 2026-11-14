"""IP-10: the CommittedDecision issuer on the real server (#371 IP-10 D1/D2)."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from suzka.r16 import decision_issuer as issuer_module
from suzka.r16.contracts import CandidateKind, ReasonCode
from suzka.r16.decision_issuer import (
    DecisionIssueStatus,
    _DecisionIssuer,
    _DecisionIssueResult,
)
from suzka.r16.history_codec import GENESIS, PolicyKind, PolicyRow
from suzka.r16.history_reader import HeadStatus, SubjectHistoryReader
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError
from suzka.r16.outcomes import R16OwnerReason
from suzka.r16.plan_preview import _PlanPreviewResult
from suzka.r16.successor_policy import (
    Interpretation,
    SuccessorPolicyInputV2,
    SuccessorRow,
    evaluate_sealed,
)
from suzka.runtime.event_journal import EventLifecycle

from test_r16_u2a_source_owner import _app, settings_in
from test_r16_u3a_plan_preview import _qualified_commitment, _qualified_goal


def _settings(root, *, history: bool = True):
    settings = settings_in(root)
    return settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            ),
            "subject_history": settings.subject_history.model_copy(
                update={
                    "enabled": history,
                    "path": root / "history" / "subject_history.sqlite3",
                }
            ),
        }
    )


@pytest.fixture
def app(tmp_path):
    return _app(_settings(tmp_path / "ip10"))


def _issue(state, goal_id):
    return state._plan_preview_owner._submit_preview(goal_id).result(timeout=30).value


def _history_transactions(state):
    return [
        item
        for item in state.event_journal.inspect().completed_transactions
        if any(
            participant == "subject.history"
            for participant, _outcome in item.participant_outcomes
        )
    ]


def _reader_head(state, issued):
    assert issued.decision is not None
    return SubjectHistoryReader(state.event_journal, state.subject_history_store).head(
        issued.decision.lineage_scope
    )


def test_issuer_is_installed_only_with_subject_history(tmp_path) -> None:
    with TestClient(_app(_settings(tmp_path / "off", history=False))) as client:
        state = client.app.state
        assert state.decision_issuer is None
        assert state._plan_preview_owner._issuer is None
        x, gid = _qualified_goal(state)
        assert isinstance(_issue(state, gid), _PlanPreviewResult)  # unchanged U3-A
    with TestClient(_app(_settings(tmp_path / "on"))) as client:
        state = client.app.state
        assert type(state.decision_issuer) is _DecisionIssuer
        assert state._plan_preview_owner._issuer is state.decision_issuer


def test_initial_decision_is_committed_and_adopted(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        before = (
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
            state.main_loop._value_system.snapshot(),
        )
        issued = _issue(state, gid)
        assert type(issued) is _DecisionIssueResult
        assert issued.status is DecisionIssueStatus.ISSUED
        assert issued.row is SuccessorRow.ROW_3
        committed = issued.decision
        assert committed is not None
        assert committed.policy_kind is PolicyKind.SUCCESSOR_POLICY_V2
        assert committed.policy_row is PolicyRow.ROW_3
        assert committed.predecessor_key == GENESIS
        selected = {c.candidate_id: c for c in committed.decision.candidates}[
            committed.decision.selected.candidate_id
        ]
        assert (selected.kind, selected.candidate_id) == (CandidateKind.PLAN, "candidate-a")
        # Sealed replay re-derives the same row (D6-3).
        assert evaluate_sealed(committed.policy_input_bytes).row is SuccessorRow.ROW_3
        sealed = SuccessorPolicyInputV2.from_bytes(committed.policy_input_bytes)
        assert sealed.interpretation is Interpretation.NOT_REVIEWED
        snapshot = state.agent_state_store.load()
        assert sealed.as_of_r07_sequence <= snapshot.last_processed_event_sequence
        # R07 transaction with the subject.history participant, completed.
        assert len(_history_transactions(state)) == 1
        head = _reader_head(state, issued)
        assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, committed, 1)
        row = state.subject_history_store.operation(committed.incarnation_id, head.transaction_id)
        assert row is not None and row.phase is HistoryPhase.FINALIZED
        # The receipt's state_before is the sealed snapshot hash.
        finalized = state.subject_history_store.finalized_decision(
            committed.incarnation_id, head.transaction_id
        )
        assert finalized is not None
        assert finalized.receipt.prepared_state_hash_before == sealed.as_of_snapshot_hash
        assert before == (
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
            state.main_loop._value_system.snapshot(),
        )
        assert state._plan_preview_owner._issuer._pending is None

        # The same request again is idempotent: no second transaction.
        again = _issue(state, gid)
        assert again.status is DecisionIssueStatus.EXISTING
        assert again.decision_id == committed.decision_id
        assert len(_history_transactions(state)) == 1


def test_commitment_acceptance_issues_the_row_4_successor(app) -> None:
    """D2 P1-1: driver's adopted -> commitment_accepted two-stage path."""

    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        first = _issue(state, gid)
        assert first.row is SuccessorRow.ROW_3
        _qualified_commitment(state, x)
        second = _issue(state, gid)
        assert second.status is DecisionIssueStatus.ISSUED
        assert second.row is SuccessorRow.ROW_4
        assert second.decision.predecessor_key == first.decision_id
        assert second.decision.predecessor_digest == first.decision.content_digest
        selected = second.decision.decision.selected.candidate_id
        assert selected == "candidate-b"
        head = _reader_head(state, second)
        assert (head.status, head.depth) == (HeadStatus.ADOPTED, 2)


def test_interpretation_change_is_a_successor_witness(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        first = _issue(state, gid)
        issuer = state.decision_issuer
        monkeypatch.setattr(
            issuer,
            "_interpretation",
            lambda *args: SimpleNamespace(outcome="EPISODE", stage="S2", status="PRESENT"),
        )
        second = _issue(state, gid)
        assert second.status is DecisionIssueStatus.ISSUED
        assert second.row is SuccessorRow.ROW_3
        assert second.decision.predecessor_key == first.decision_id
        sealed = SuccessorPolicyInputV2.from_bytes(second.decision.policy_input_bytes)
        assert sealed.interpretation is Interpretation.SETTLED


def test_contradictory_interpretation_holds_without_a_transaction(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        monkeypatch.setattr(
            state.decision_issuer,
            "_interpretation",
            lambda *args: SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTRADICTORY"),
        )
        held = _issue(state, gid)
        assert held.status is DecisionIssueStatus.HOLD
        assert held.row is SuccessorRow.ROW_2B
        assert held.hold is not None and held.hold.reason is ReasonCode.CONFLICTING_EVIDENCE
        assert _history_transactions(state) == []
        assert state.subject_history_store.has_operations() is False


def test_unavailable_interpretation_is_row_1(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)

        def broken(*args):
            raise RuntimeError("R15 unavailable")

        monkeypatch.setattr(state.decision_issuer, "_interpretation", broken)
        held = _issue(state, gid)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_1)
        assert _history_transactions(state) == []


def test_source_drift_without_new_witness_holds(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        _issue(state, gid)
        real = issuer_module._context_digest
        monkeypatch.setattr(
            issuer_module, "_context_digest", lambda q: "e" * 64 if real(q) else real(q)
        )
        held = _issue(state, gid)
        assert held.status is DecisionIssueStatus.HOLD
        assert held.owner_reason is R16OwnerReason.DECISION_SOURCE_DRIFT
        assert len(_history_transactions(state)) == 1


def test_capacity_and_slot_refusals_hold_before_prepare(app, monkeypatch) -> None:
    """D2 P1-3: neither refusal fail-stops the runtime."""

    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        store = state.subject_history_store

        def full(length):
            raise HistoryStoreCapacityError("full")

        with monkeypatch.context() as patch:
            patch.setattr(store, "check_headroom", full)
            held = _issue(state, gid)
        assert (held.status, held.owner_reason) == (
            DecisionIssueStatus.HOLD,
            R16OwnerReason.HISTORY_CAPACITY,
        )
        assert store.has_operations() is False
        # The runtime is healthy: the next issue succeeds.
        assert _issue(state, gid).status is DecisionIssueStatus.ISSUED


def test_unverifiable_head_is_a_conflicting_predecessor(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        _issue(state, gid)
        monkeypatch.setattr(state.decision_issuer, "_head_verified", lambda *args: False)
        held = _issue(state, gid)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_0)


def test_unqualified_goal_still_yields_a_policy_row(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state, adopt=False)
        held = _issue(state, gid)
        assert held.status is DecisionIssueStatus.HOLD
        assert held.row in (SuccessorRow.ROW_0, SuccessorRow.ROW_1)
        assert _history_transactions(state) == []


def test_validate_commit_requires_exactly_its_own_prepared_row(app) -> None:
    """D2 P1-2: the pre-WAL check sees its own PREPARED row only."""

    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        issuer = state.decision_issuer
        seen: list[str] = []
        real = issuer._validate_commit

        def spy(event):
            pending = issuer._pending
            assert pending is not None
            rows = state.subject_history_store.operations(pending.incarnation)
            seen.extend(row.phase.value for row in rows)
            return real(event)

        issuer._validate_commit = spy  # type: ignore[method-assign]
        try:
            issued = _issue(state, gid)
        finally:
            issuer._validate_commit = real  # type: ignore[method-assign]
        assert issued.status is DecisionIssueStatus.ISSUED
        assert seen == [HistoryPhase.PREPARED.value]


def test_decision_survives_restart_and_stays_idempotent(tmp_path) -> None:
    settings = _settings(tmp_path / "restart")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        issued = _issue(state, gid)
        assert issued.status is DecisionIssueStatus.ISSUED
    with TestClient(_app(settings)) as client:
        state = client.app.state
        head = SubjectHistoryReader(state.event_journal, state.subject_history_store).head(
            issued.decision.lineage_scope
        )
        assert (head.status, head.decision) == (HeadStatus.ADOPTED, issued.decision)
        again = _issue(state, gid)
        assert again.status is DecisionIssueStatus.EXISTING
        assert again.decision_id == issued.decision_id


def test_hold_events_complete_without_failure(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        monkeypatch.setattr(
            state.decision_issuer,
            "_interpretation",
            lambda *args: SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTRADICTORY"),
        )
        _issue(state, gid)
        records = state.event_journal.inspect().records
        assert not any(record.lifecycle is EventLifecycle.FAILED for record in records)
