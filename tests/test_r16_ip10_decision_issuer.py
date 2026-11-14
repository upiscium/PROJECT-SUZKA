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


def _startup_wired(state) -> None:
    """Review P2-4: the store, reconciliation and compaction agree."""

    assert state.subject_history_store is not None
    assert state.startup_reconciliation.history_store is state.subject_history_store
    assert state.journal_compaction.history_attestation is not None


def test_fresh_and_existing_journal_startups_are_wired(tmp_path) -> None:
    settings = _settings(tmp_path / "wired")
    with TestClient(_app(settings)) as client:  # brand-new Journal: deferred
        _startup_wired(client.app.state)
    with TestClient(_app(settings)) as client:  # existing Journal: direct
        _startup_wired(client.app.state)


def test_row_4b_and_existing_after_commitment(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        _qualified_commitment(state, x)
        monkeypatch.setattr(
            state.decision_issuer,
            "_interpretation",
            lambda *args: SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTESTED"),
        )
        issued = _issue(state, gid)
        assert (issued.status, issued.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_4B)
        assert issued.decision.decision.selected.candidate_id == "candidate-a"
        again = _issue(state, gid)
        assert (again.status, again.decision_id) == (
            DecisionIssueStatus.EXISTING,
            issued.decision_id,
        )


def test_occupied_lineage_slot_holds_before_prepare(app, monkeypatch) -> None:
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
        # A successor that would reuse the GENESIS slot of the first Decision.
        monkeypatch.setattr(issuer, "_committed", lambda *args: first.decision)
        held = _issue(state, gid)
        assert (held.status, held.owner_reason) == (
            DecisionIssueStatus.HOLD,
            R16OwnerReason.DECISION_SLOT_OCCUPIED,
        )
        assert len(_history_transactions(state)) == 1


def test_deadline_margin_holds_before_prepare(app, monkeypatch) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        commitment_id = _qualified_commitment(state, x)
        record = state.main_loop._commitment_system.get(commitment_id)
        at = record.deadline.at
        if at is None:
            pytest.skip("fixture Commitment has no deadline")
        issuer = state.decision_issuer
        monkeypatch.setattr(issuer, "_clock", lambda: at - issuer_module.DEADLINE_MARGIN / 2)
        held = _issue(state, gid)
        assert (held.status, held.owner_reason) == (
            DecisionIssueStatus.HOLD,
            R16OwnerReason.DEADLINE_ELAPSED,
        )
        monkeypatch.setattr(issuer, "_clock", lambda: at - issuer_module.DEADLINE_MARGIN * 2)
        assert _issue(state, gid).status is DecisionIssueStatus.ISSUED


def test_head_without_finalized_proof_is_a_conflict(app, monkeypatch) -> None:
    """Review P2-2: real head check, not a patched verifier."""

    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        _issue(state, gid)
        monkeypatch.setattr(
            state.subject_history_store, "finalized_decision", lambda *args: None
        )
        held = _issue(state, gid)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_0)


def test_store_read_failure_is_a_hold_not_a_failed_event(app, monkeypatch) -> None:
    """Review P2-3."""

    from suzka.r16.history_store import HistoryStoreUnavailable

    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)

        def unavailable(*args):
            raise HistoryStoreUnavailable("locked")

        with monkeypatch.context() as patch:
            patch.setattr(state.subject_history_store, "operations", unavailable)
            held = _issue(state, gid)
        # The reader turns it into an unavailable lineage (row 1); any later
        # read failure is an issuer HOLD.  Never a FAILED event.
        assert held.status is DecisionIssueStatus.HOLD
        assert held.row is SuccessorRow.ROW_1 or (
            held.owner_reason is R16OwnerReason.SOURCE_UNAVAILABLE
        )
        records = state.event_journal.inspect().records
        assert not any(record.lifecycle is EventLifecycle.FAILED for record in records)
        assert _issue(state, gid).status is DecisionIssueStatus.ISSUED


def test_crash_after_prepare_is_tombstoned_then_reissued(tmp_path) -> None:
    """Review P2-1: a fail-stop at _validate_commit recovers at startup."""

    from suzka.runtime.agent_runtime import AgentRuntimeDurabilityError

    settings = _settings(tmp_path / "crash-prepare")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        issuer = state.decision_issuer

        def crash(event):
            raise RuntimeError("simulated pre-WAL failure")

        issuer._validate_commit = crash  # type: ignore[method-assign]
        with pytest.raises(AgentRuntimeDurabilityError):
            _issue(state, gid)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        rows = state.subject_history_store.operations(
            state.decision_issuer._pending.incarnation
            if state.decision_issuer._pending
            else _incarnation(state)
        )
        assert [row.phase for row in rows] == [HistoryPhase.ABORTED]
        issued = _issue(state, gid)
        assert issued.status is DecisionIssueStatus.ISSUED


def test_crash_after_internal_commit_rolls_forward_then_existing(tmp_path) -> None:
    """Review P2-1: a failure after R06 is finalized at startup."""

    from suzka.r16.history_participant import SubjectHistoryParticipant
    from suzka.runtime.agent_runtime import AgentRuntimeDurabilityError

    settings = _settings(tmp_path / "crash-finalize")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        real = SubjectHistoryParticipant.finalize

        def crash(self, binding):
            raise RuntimeError("simulated post-R06 failure")

        SubjectHistoryParticipant.finalize = crash  # type: ignore[method-assign]
        try:
            with pytest.raises(AgentRuntimeDurabilityError):
                _issue(state, gid)
        finally:
            SubjectHistoryParticipant.finalize = real  # type: ignore[method-assign]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        rows = state.subject_history_store.operations(_incarnation(state))
        assert [row.phase for row in rows] == [HistoryPhase.FINALIZED]
        again = _issue(state, gid)
        assert again.status is DecisionIssueStatus.EXISTING


def _incarnation(state):
    from suzka.r16.history_adoption import current_incarnation

    return current_incarnation(state.event_journal, state.subject_history_store)
