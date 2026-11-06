"""Bounded R16 U2b in-process MVI tests with disposable committed sources."""

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from suzka.memory.experience_participant import experience_id_for_event
from suzka.motivation.common import R13ReferenceKind
from suzka.motivation.goal import GoalLifecycle
from suzka.r16.goal_owner import (
    _GoalSourceOwner,
    _GoalSourceSelection,
    _GoalSourceUnavailable,
)
from suzka.r16.source_owner import _MotivationSourceRef
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeExecutionError,
)
from suzka.runtime.context import ContextStatus

from test_r16_u2a_source_owner import _app, settings_in


@pytest.fixture
def app(tmp_path):
    settings = settings_in(tmp_path / "goal")
    settings = settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            )
        }
    )
    return _app(settings)


def chat(state, text):
    result = state.agent_runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: state.main_loop.chat(text),
    ).result(timeout=30)
    return _MotivationSourceRef(
        result.event.event_id,
        result.event.processing_sequence,
        result.value.episode_id,
    )


def x_id(source):
    return experience_id_for_event(source.event_id, source.event_sequence)


def preexisting_interest(state, source):
    before = state.main_loop.export_goal_state()
    result = state._motivation_source_owner._submit(source).result(timeout=30)
    assert result.value == "motivation_evidence_applied"
    assert len(state.main_loop.export_motivation_state().records) == 1
    assert state.main_loop.export_goal_state() == before
    return result


def setup_sources(state, *, with_interest=True, request=None):
    x = chat(state, "initial subjective experience; test fixture only")
    if with_interest:
        m = preexisting_interest(state, x)
    else:
        m = None
    p = chat(
        state,
        request if request is not None else f"REVIEW_EXPERIENCE:{x_id(x)}",
    )
    return x, m, p


def test_real_serialized_proposal_adoption_persisted_without_cognition_replay(app):
    settings = app.state.settings
    with TestClient(app) as client:
        state = client.app.state
        assert isinstance(state._goal_source_owner, _GoalSourceOwner)
        x, m, p = setup_sources(state)
        # Unrelated default care/honesty Value tension is not evidence that
        # this particular Experience review conflicts with a Value.
        assert state.main_loop._value_system.snapshot().conflicts
        before_commitments = state.main_loop.export_commitment_state()
        model_calls = state.model_provider.calls
        selection = _GoalSourceSelection(p, x)
        proposed = state._goal_source_owner._submit_proposal(selection).result(
            timeout=30
        )
        goal_id = proposed.value
        first_goal = state.main_loop._goal_system.get(goal_id)
        assert first_goal is not None
        assert first_goal.lifecycle is GoalLifecycle.PROPOSED
        assert first_goal.target.kind is R13ReferenceKind.EXPERIENCE
        assert first_goal.target.reference == x_id(x)
        assert (
            tuple(item.kind for item in first_goal.evidence_refs).count(
                R13ReferenceKind.MOTIVATION
            )
            == 1
        )
        assert any(
            item.kind is R13ReferenceKind.EVENT and item.reference == m.event.event_id
            for item in first_goal.evidence_refs
        )
        assert first_goal.subject_admission is None
        assert (
            proposed.event.processing_sequence
            > p.event_sequence
            > m.event.processing_sequence
        )
        assert state.model_provider.calls == model_calls
        assert state.main_loop.export_commitment_state() == before_commitments
        review = state._goal_source_owner._submit_review(goal_id).result(timeout=30)
        assert review.value == "adopted"
        adopted = state.main_loop._goal_system.get(goal_id)
        assert adopted is not None and adopted.lifecycle is GoalLifecycle.ADOPTED
        assert adopted.subject_admission is not None
        assert adopted.subject_admission.event_id == review.event.event_id
        assert review.event.processing_sequence > proposed.event.processing_sequence
        assert state.model_provider.calls == model_calls
        assert state.main_loop.export_commitment_state() == before_commitments
        saved = state.agent_state_store.load()
        assert saved.schema_version == 10
        assert saved.r13_state.goal.restore() == state.main_loop.export_goal_state()
        assert state._goal_source_owner._pending is None

    with TestClient(_app(settings)) as client:
        state = client.app.state
        restored = state.main_loop._goal_system.get(goal_id)
        assert restored == adopted
        assert state._goal_source_owner._pending is None
        model_calls = state.model_provider.calls
        replay = state._goal_source_owner._submit_review(goal_id).result(timeout=30)
        assert replay.value == "hold"
        assert state.main_loop._goal_system.get(goal_id) == adopted
        assert state.model_provider.calls == model_calls


@pytest.mark.parametrize("case", ["no_motivation", "bad_grammar", "wrong_target"])
def test_source_unqualified_fails_before_proposal(app, case):
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "initial subjective experience; test fixture only")
        if case != "no_motivation":
            preexisting_interest(state, x)
        request = (
            "free text is not an R16 Goal"
            if case == "bad_grammar"
            else f"REVIEW_EXPERIENCE:{x_id(x)}"
        )
        p = chat(state, request)
        selector = _GoalSourceSelection(p, x)
        if case == "wrong_target":
            other = chat(state, "unrelated separate target")
            selector = _GoalSourceSelection(p, other)
        before = state.main_loop.export_goal_state()
        with pytest.raises(AgentRuntimeExecutionError):
            state._goal_source_owner._submit_proposal(selector).result(timeout=30)
        assert state.main_loop.export_goal_state() == before
        assert state._goal_source_owner._pending is None


def test_independent_review_holds_after_source_context_loss(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = setup_sources(state)
        goal_id = (
            state._goal_source_owner._submit_proposal(_GoalSourceSelection(p, x))
            .result(timeout=30)
            .value
        )
        goal = state.main_loop._goal_system.get(goal_id)
        registry = state.main_loop.context_registry
        original = registry.get
        with monkeypatch.context() as patch:
            patch.setattr(
                registry,
                "get",
                lambda key: replace(original(key), status=ContextStatus.SUSPENDED),
            )
            result = state._goal_source_owner._submit_review(goal_id).result(timeout=30)
            assert result.value == "hold"
        assert state.main_loop._goal_system.get(goal_id) == goal
        assert state._goal_source_owner._pending is None


def test_pre_wal_source_change_aborts_proposal_and_restores_r13(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = setup_sources(state)
        owner = state._goal_source_owner
        original = owner._qualify
        calls = 0

        def qualify(selection, event):
            nonlocal calls
            calls += 1
            result = original(selection, event)
            if calls == 2:
                return replace(result, target_record_digest="0" * 64)
            return result

        before = state.main_loop.export_goal_state()
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_qualify", qualify)
            with pytest.raises(Exception):
                owner._submit_proposal(_GoalSourceSelection(p, x)).result(timeout=30)
        assert calls >= 2
        assert state.main_loop.export_goal_state() == before
        assert state._goal_source_owner._pending is None


def test_unbound_owner_or_forged_selection_does_not_mutate(app):
    with TestClient(app) as client:
        state = client.app.state
        before = state.main_loop.export_goal_state()
        owner = state._goal_source_owner
        with pytest.raises(_GoalSourceUnavailable):
            owner._submit_proposal("caller-authored proof")
        with pytest.raises(_GoalSourceUnavailable):
            state.main_loop._propose_goal_source(owner, object())
        assert state.main_loop.export_goal_state() == before
