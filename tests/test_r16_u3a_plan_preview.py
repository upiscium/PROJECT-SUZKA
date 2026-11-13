"""R16 U3-A private source-qualified Plan preview, not durable Decision authority."""

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from suzka.motivation.goal import GoalLifecycle
from suzka.r16.outcomes import R16OwnerStatus
from suzka.r16.commitment_owner import _CommitmentSourceSelection
from suzka.r16.contracts import (
    CandidateKind,
    DecisionRecord,
    ReasonCode,
    StepKind,
    UNCERTAINTY_SCALE,
)
from suzka.r16.goal_owner import _GoalSourceSelection
from suzka.r16.plan_preview import (
    _ASSESS,
    _COMPARE,
    _PLAN_A,
    _PLAN_B,
    _VERIFY,
    _PlanPreviewOwner,
    _PlanSourceUnavailable,
)
from suzka.runtime.agent_runtime import AgentRuntimeDurabilityError
from suzka.runtime.context import ContextStatus

from test_r16_u2a_source_owner import _app, settings_in
from test_r16_u2b_goal_owner import chat, preexisting_interest, x_id


@pytest.fixture
def app(tmp_path):
    settings = settings_in(tmp_path / "u3a-preview")
    settings = settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            ),
        }
    )
    return _app(settings)


def _qualified_goal(state, *, adopt: bool = True):
    x = chat(state, "source experience; fixture")
    preexisting_interest(state, x)
    p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
    owner = state._goal_source_owner
    proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(timeout=30)
    if adopt:
        adopted = owner._submit_review(proposed.value.subject_id).result(timeout=30)
        assert adopted.value.status is R16OwnerStatus.ADOPTED
    return x, proposed.value.subject_id


def _qualified_commitment(state, x):
    p = chat(state, f"COMMIT_REVIEW_EXPERIENCE:1:{x_id(x)}")
    owner = state._commitment_source_owner
    proposal = owner._submit_proposal(_CommitmentSourceSelection(p, x)).result(
        timeout=30
    )
    accepted = owner._submit_review(proposal.value.subject_id).result(timeout=30)
    assert accepted.value.status is R16OwnerStatus.ACCEPTED
    return proposal.value.subject_id


@pytest.mark.parametrize("with_commitment", (False, True))
def test_real_committed_source_selects_distinct_closed_plan_without_effect(
    app, with_commitment
):
    settings = app.state.settings
    with TestClient(app) as client:
        state = client.app.state
        owner = state._plan_preview_owner
        assert type(owner) is _PlanPreviewOwner
        x, gid = _qualified_goal(state)
        commitment_id = _qualified_commitment(state, x) if with_commitment else None
        before_roots = (
            state.main_loop.export_goal_state(),
            state.main_loop.export_motivation_state(),
            state.main_loop.export_commitment_state(),
            state.main_loop._value_system.snapshot(),
        )
        model_calls = state.model_provider.calls

        result = owner._submit_preview(gid).result(timeout=30)
        preview = result.value
        assert preview.is_authoritative is False
        assert preview.hold is None
        assert isinstance(preview.decision, DecisionRecord)
        assert preview.witness is not None
        record = preview.decision
        witness = preview.witness
        expected_branch = _PLAN_B if with_commitment else _PLAN_A
        assert witness.selected_plan == expected_branch
        assert witness.qualification.commitment_id == commitment_id
        assert witness.qualification.value_id == "honesty"
        assert witness.norm == "human-approved-source-review-honesty/v1"
        assert witness.no_op_eligible is False
        assert all(observation.satisfied for observation in witness.observations)
        assert tuple(item.step_id for item in witness.observations) == (
            _VERIFY,
            _COMPARE,
            _ASSESS,
        )
        assert len(record.candidates) == 3
        assert tuple(c.kind for c in record.candidates) == (
            CandidateKind.PLAN,
            CandidateKind.PLAN,
            CandidateKind.NO_OP,
        )
        a, b, noop = record.candidates
        assert len(a.plan.steps) == 2
        assert len(b.plan.steps) == 3
        assert noop.plan is None
        assert [(s.step_id, s.kind, s.depends_on) for s in a.plan.steps] == [
            (_VERIFY, StepKind.CHECK, ()),
            (_ASSESS, StepKind.ASSESS, (_VERIFY,)),
        ]
        assert [(s.step_id, s.kind, s.depends_on) for s in b.plan.steps] == [
            (_VERIFY, StepKind.CHECK, ()),
            (_COMPARE, StepKind.COMPARE, (_VERIFY,)),
            (_ASSESS, StepKind.ASSESS, (_COMPARE,)),
        ]
        expected_choice = b if with_commitment else a
        assert record.selected.candidate_id == expected_choice.candidate_id
        assert record.selected.candidate_digest == expected_choice.content_digest
        assert record.reason is ReasonCode.PLAN_PREFERRED
        assert record.uncertainty == UNCERTAINTY_SCALE
        assert len(record.canonical_bytes()) < 65_536
        assert DecisionRecord.from_json(record.canonical_bytes()) == record
        assert record.explanation().reason is ReasonCode.PLAN_PREFERRED
        assert state.model_provider.calls == model_calls
        assert before_roots == (
            state.main_loop.export_goal_state(),
            state.main_loop.export_motivation_state(),
            state.main_loop.export_commitment_state(),
            state.main_loop._value_system.snapshot(),
        )
        assert state.agent_state_store.load().schema_version == 10
        assert owner._pending is owner._inflight is None

    # Technical event / Journal changes are durable; the Decision value is not.
    with TestClient(_app(settings)) as client:
        recovered = client.app.state
        assert recovered._plan_preview_owner._pending is None
        assert recovered._plan_preview_owner._inflight is None
        assert recovered.main_loop._goal_system.get(gid).lifecycle is (
            GoalLifecycle.ADOPTED
        )
        assert not hasattr(recovered.agent_state_store.load(), "decision_state")
        # Reconstructing a new preview is permitted, not WAL decision replay.
        repeat = recovered._plan_preview_owner._submit_preview(gid).result(timeout=30)
        assert repeat.value.decision is not None
        assert repeat.value.witness.selected_plan == expected_branch


@pytest.mark.parametrize("case", ("no_goal", "unadopted_goal", "inactive_context"))
def test_unqualified_goal_or_context_holds_without_mutating_r13(app, case, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        gid = "a" * 64
        if case != "no_goal":
            x, gid = _qualified_goal(state, adopt=case != "unadopted_goal")
        before = (
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
            state.main_loop._value_system.snapshot(),
        )
        registry = state.main_loop.context_registry
        with monkeypatch.context() as patch:
            if case == "inactive_context":
                original = registry.get
                patch.setattr(
                    registry,
                    "get",
                    lambda ctx: replace(original(ctx), status=ContextStatus.SUSPENDED),
                )
            result = state._plan_preview_owner._submit_preview(gid).result(timeout=30)
        assert result.value.decision is None
        assert result.value.hold is not None
        assert result.value.hold.reason is ReasonCode.INSUFFICIENT_EVIDENCE
        assert before == (
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
            state.main_loop._value_system.snapshot(),
        )


def test_missing_review_integrity_value_holds(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        owner = state._plan_preview_owner
        with monkeypatch.context() as patch:
            patch.setattr(
                owner,
                "_qualify_value",
                lambda ctx: (_ for _ in ()).throw(
                    _PlanSourceUnavailable("seed value not qualified")
                ),
            )
            result = owner._submit_preview(gid).result(timeout=30)
        assert result.value.decision is None
        assert result.value.hold.reason is ReasonCode.INSUFFICIENT_EVIDENCE


def test_context_revision_change_during_preview_returns_hold(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        owner = state._plan_preview_owner
        old = owner._qualify
        calls = 0

        def qualify(gid, event):
            nonlocal calls
            calls += 1
            value = old(gid, event)
            if calls == 2:
                return replace(
                    value,
                    context_last_modified_revision=value.context_last_modified_revision
                    + 1,
                )
            return value

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_qualify", qualify)
            outcome = owner._submit_preview(gid).result(timeout=30)
        assert calls >= 2
        assert outcome.value.hold is not None
        assert outcome.value.decision is None


def test_current_context_source_changed_before_wal_rejects_preview(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        owner = state._plan_preview_owner
        old = owner._qualify
        calls = 0

        def qualify(gid, event):
            nonlocal calls
            calls += 1
            value = old(gid, event)
            if calls == 3:
                return replace(value, value_digest="0" * 64)
            return value

        before = state.main_loop.export_goal_state()
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_qualify", qualify)
            with pytest.raises(AgentRuntimeDurabilityError):
                owner._submit_preview(gid).result(timeout=30)
        assert calls >= 3
        assert state.main_loop.export_goal_state() == before
        assert owner._pending is None


def test_untrusted_caller_cannot_present_plan_or_nonce(app):
    with TestClient(app) as client:
        state = client.app.state
        owner = state._plan_preview_owner
        before = state.main_loop.export_goal_state()
        with pytest.raises((TypeError, ValueError)):
            owner._submit_preview("forged Plan authority or proof")
        with pytest.raises(_PlanSourceUnavailable):
            state.main_loop._review_plan_preview(owner, object())
        assert state.main_loop.export_goal_state() == before


def test_global_unrelated_value_conflict_does_not_block_goal_only_plan(app):
    with TestClient(app) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        assert state.main_loop._value_system.snapshot().conflicts
        preview = (
            state._plan_preview_owner._submit_preview(gid).result(timeout=30).value
        )
        assert preview.decision is not None
        assert preview.witness.selected_plan == _PLAN_A


def test_proposed_but_unaccepted_responsibility_is_not_goal_only_plan(app):
    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        request = chat(state, f"COMMIT_REVIEW_EXPERIENCE:1:{x_id(x)}")
        proposed = state._commitment_source_owner._submit_proposal(
            _CommitmentSourceSelection(request, x)
        ).result(timeout=30)
        assert proposed.value
        roots = (
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
        )
        preview = (
            state._plan_preview_owner._submit_preview(gid).result(timeout=30).value
        )
        assert preview.decision is None
        assert preview.hold.reason is ReasonCode.INSUFFICIENT_EVIDENCE
        assert roots == (
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
        )


def test_expired_commitment_cannot_be_misreported_as_absent(app, monkeypatch):
    from datetime import timedelta

    import suzka.r16.commitment_owner as commitment_module

    with TestClient(app) as client:
        state = client.app.state
        x, gid = _qualified_goal(state)
        cid = _qualified_commitment(state, x)
        deadline = state.main_loop._commitment_system.get(cid).deadline.at
        assert deadline is not None
        with monkeypatch.context() as patch:
            patch.setattr(
                commitment_module,
                "_utcnow",
                lambda: deadline + timedelta(seconds=1),
            )
            preview = (
                state._plan_preview_owner._submit_preview(gid).result(timeout=30).value
            )
        assert preview.decision is None
        assert preview.hold.reason is ReasonCode.INSUFFICIENT_EVIDENCE


def test_adoption_event_provenance_is_required(app, monkeypatch):
    from suzka.runtime.agent_runtime import AgentEventType

    with TestClient(app) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        owner = state._plan_preview_owner
        terminal = owner._require_terminal

        def require_terminal(event_id, event_sequence, kind, source):
            if kind is AgentEventType.GOAL_ADOPTION_REVIEW:
                raise _PlanSourceUnavailable("missing trusted adoption Journal")
            return terminal(event_id, event_sequence, kind, source)

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_require_terminal", require_terminal)
            preview = owner._submit_preview(gid).result(timeout=30).value
        assert preview.decision is None
        assert preview.hold.reason is ReasonCode.INSUFFICIENT_EVIDENCE


def test_value_revision_change_during_preview_holds(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        owner = state._plan_preview_owner
        original = owner._qualify_value
        reads = 0

        def qualify_value(context_id):
            nonlocal reads
            reads += 1
            revision, digest, origin = original(context_id)
            if reads == 2:
                return revision + 1, digest, origin
            return revision, digest, origin

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_qualify_value", qualify_value)
            preview = owner._submit_preview(gid).result(timeout=30).value
        assert reads >= 2
        assert preview.decision is None
        assert preview.hold.reason is ReasonCode.INSUFFICIENT_EVIDENCE


def test_missing_honesty_seed_does_not_become_fabricated_source(tmp_path):
    settings = settings_in(tmp_path / "no-honesty")
    seeds = tuple(item for item in settings.values.seeds if item.value_id != "honesty")
    assert len(seeds) != len(settings.values.seeds)
    settings = settings.model_copy(
        update={
            "values": settings.values.model_copy(
                update={"seeds": seeds, "conflicts": ()}
            ),
            "emotion": settings.emotion.model_copy(update={"timer_enabled": False}),
        }
    )
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        result = state._plan_preview_owner._submit_preview(gid).result(timeout=30)
        assert result.value.decision is None
        assert result.value.hold.reason is ReasonCode.INSUFFICIENT_EVIDENCE


def test_real_context_transition_during_preview_fails_closed_before_wal(
    app, monkeypatch
):
    """An actual ContextRegistry mutation is never a valid preview receipt."""
    with TestClient(app) as client:
        state = client.app.state
        _, gid = _qualified_goal(state)
        owner = state._plan_preview_owner
        registry = state.main_loop.context_registry
        original = owner._build_decision
        initial_context = registry.state
        initial_goal = state.main_loop.export_goal_state()
        initial_commitment = state.main_loop.export_commitment_state()
        changed = False

        def build_with_source_transition(q):
            nonlocal changed
            candidate = original(q)
            # Deliberate fault injection, NOT a legitimate R09 writer:
            # mutate the actual Context authority during preview evaluation.
            registry.suspend(q.context_id)
            changed = True
            return candidate

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_build_decision", build_with_source_transition)
            with pytest.raises(AgentRuntimeDurabilityError):
                owner._submit_preview(gid).result(timeout=30)

        assert changed
        # Pre-WAL snapshot restore returns every authority to last committed
        # state; there is no Decision receipt, returned value or Action.
        assert registry.state == initial_context
        assert state.main_loop.export_goal_state() == initial_goal
        assert state.main_loop.export_commitment_state() == initial_commitment
        assert state.agent_state_store.load().schema_version == 10
        assert owner._pending is None
