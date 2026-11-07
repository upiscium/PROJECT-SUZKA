"""Scoped U2c proof: real serialized R07/R12/DB1 with synthetic provider.

This does not create a tool, verified Outcome or R19 scheduler. "Acceptance"
is a source-checked bounded prospective obligation, NOT a completed deliverable.
"""

from dataclasses import replace
from datetime import timedelta

from fastapi.testclient import TestClient
import pytest

import suzka.r16.commitment_owner as commitment_owner_module
from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.commitment import CommitmentLifecycle
from suzka.r16.commitment_owner import (
    _CommitmentSourceOwner,
    _CommitmentSourceSelection,
    _CommitmentSourceUnavailable,
)
from suzka.r16.source_owner import _MotivationSourceRef
from suzka.memory.experience_participant import experience_id_for_event
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeExecutionError,
)
from suzka.runtime.chat_context import ChatContextSelectors
from suzka.runtime.context import ContextStatus

from test_r16_u2a_source_owner import _app, settings_in


@pytest.fixture
def app(tmp_path):
    settings = settings_in(tmp_path / "commitment")
    settings = settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            ),
        }
    )
    return _app(settings)


def _chat(state, message, selectors=None):
    result = state.agent_runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: state.main_loop.chat(message, selectors),
    ).result(timeout=30)
    return _MotivationSourceRef(
        result.event.event_id,
        result.event.processing_sequence,
        result.value.episode_id,
    )


def _experience_id(source):
    return experience_id_for_event(source.event_id, source.event_sequence)


def _sources(state, *, motive=True, text=None, selectors=None):
    x = _chat(state, "source committed experience: synthetic")
    if motive:
        m = state._motivation_source_owner._submit(x).result(timeout=30)
        assert m.value == "motivation_evidence_applied"
    else:
        m = None
    p = _chat(
        state,
        text if text is not None else f"COMMIT_REVIEW_EXPERIENCE:1:{_experience_id(x)}",
        selectors,
    )
    return x, m, p


def _selector(p, x):
    return _CommitmentSourceSelection(p, x)


def test_real_source_separate_responsibility_accept_recovery_and_no_goal_adoption(app):
    settings = app.state.settings
    with TestClient(app) as client:
        state = client.app.state
        assert isinstance(state._commitment_source_owner, _CommitmentSourceOwner)
        x, motive, p = _sources(state)
        owner = state._commitment_source_owner
        previous_goals = state.main_loop.export_goal_state()
        calls = state.model_provider.calls
        selected = _selector(p, x)
        proposal = owner._submit_proposal(selected).result(timeout=30)
        commitment_id = proposal.value
        commitment = state.main_loop._commitment_system.get(commitment_id)
        assert commitment is not None
        assert commitment.lifecycle is CommitmentLifecycle.PROPOSED
        assert commitment.subject == "Produce a source-checked review record"
        assert commitment.beneficiary == R13Reference(
            R13ReferenceKind.SUBJECT, "subject:self"
        )
        assert commitment.scope == (
            "record_review_explanation",
            "record_source_qualification",
        )
        assert commitment.deadline.is_set
        p_record = state.experience_store.load_committed_current(
            _experience_id(p)
        ).record
        assert commitment.deadline.at == p_record.created_at + timedelta(days=7)
        assert commitment.related_goal_refs == ()
        assert commitment.desire_refs == ()
        assert commitment.subject_admission is None
        assert tuple(
            item.reference
            for item in commitment.evidence_refs
            if item.kind is R13ReferenceKind.EVENT
        ) == (motive.event.event_id,)
        assert p.event_sequence > motive.event.processing_sequence > x.event_sequence
        assert state.main_loop.export_goal_state() == previous_goals
        assert state.model_provider.calls == calls
        review = owner._submit_review(commitment_id).result(timeout=30)
        assert review.value == "accepted"
        accepted = state.main_loop._commitment_system.get(commitment_id)
        assert accepted is not None and accepted.lifecycle is CommitmentLifecycle.ACTIVE
        assert accepted.subject_admission is not None
        assert accepted.subject_admission.event_id == review.event.event_id
        assert accepted.subject_admission.deadline == commitment.deadline
        assert accepted.subject_admission.beneficiary == commitment.beneficiary
        assert accepted.subject_admission.scope == commitment.scope
        assert state.main_loop.export_goal_state() == previous_goals
        assert state.model_provider.calls == calls
        assert owner._inflight is owner._pending is None
        persisted = state.agent_state_store.load()
        assert persisted.schema_version == 10
        assert (
            persisted.r13_state.commitment.restore()
            == state.main_loop.export_commitment_state()
        )

    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.main_loop._commitment_system.get(commitment_id) == accepted
        calls = state.model_provider.calls
        result = state._commitment_source_owner._submit_review(commitment_id).result(
            timeout=30
        )
        assert result.value == "hold"
        assert state.main_loop._commitment_system.get(commitment_id) == accepted
        assert state.model_provider.calls == calls


@pytest.mark.parametrize(
    "case",
    (
        "missing_motivation",
        "malformed",
        "wrong_target",
        "later_motivation",
        "wrong_context",
    ),
)
def test_wrong_or_unqualified_source_fails_closed_without_commitment(app, case):
    with TestClient(app) as client:
        state = client.app.state
        x = _chat(state, "source committed experience: synthetic")
        if case not in ("missing_motivation", "later_motivation"):
            state._motivation_source_owner._submit(x).result(timeout=30)
        message = (
            "COMMIT_REVIEW_EXPERIENCE:1:" + _experience_id(x)
            if case != "malformed"
            else f"please commit to reviewing {_experience_id(x)}"
        )
        selectors = (
            ChatContextSelectors(client_session_id="other-session")
            if case == "wrong_context"
            else None
        )
        p = _chat(state, message, selectors)
        if case == "later_motivation":
            state._motivation_source_owner._submit(x).result(timeout=30)
        if case == "wrong_target":
            target = _chat(state, "another experience")
        else:
            target = x
        snapshot = state.main_loop.export_commitment_state()
        with pytest.raises(AgentRuntimeExecutionError):
            state._commitment_source_owner._submit_proposal(
                _selector(p, target)
            ).result(timeout=30)
        assert state.main_loop.export_commitment_state() == snapshot
        assert state._commitment_source_owner._pending is None


def test_expired_deadline_cannot_be_accepted_after_seven_days(app):
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = _sources(state)
        selected = _selector(p, x)
        reference = (
            state.agent_runtime.submit(
                AgentEventType.EMOTION_TICK,
                AgentEventSource.RUNTIME_EMOTION_TIMER,
                lambda: "no mutation",
            )
            .result(timeout=30)
            .event
        )
        due = state.experience_store.load_committed_current(
            _experience_id(p)
        ).record.created_at + timedelta(days=7)
        with pytest.raises(_CommitmentSourceUnavailable):
            state._commitment_source_owner._qualify(
                selected, replace(reference, requested_at=due + timedelta(seconds=1))
            )
        assert not state.main_loop.export_commitment_state().records


def test_pre_deadline_queued_review_holds_when_runtime_clock_has_expired(
    app, monkeypatch
):
    """A queued event's requested_at must not grant a post-deadline admission."""
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = _sources(state)
        owner = state._commitment_source_owner
        cid = owner._submit_proposal(_selector(p, x)).result(timeout=30).value
        initial = state.main_loop.export_commitment_state()
        due = state.main_loop._commitment_system.get(cid).deadline.at
        assert due is not None
        with monkeypatch.context() as patch:
            # The R07 event is genuinely submitted before the due date.
            # The independent trusted processing-time clock is after due.
            patch.setattr(
                commitment_owner_module,
                "_utcnow",
                lambda: due + timedelta(seconds=1),
            )
            outcome = owner._submit_review(cid).result(timeout=30)
        assert outcome.event.requested_at < due
        assert outcome.value == "hold"
        assert state.main_loop.export_commitment_state() == initial
        assert state.main_loop._commitment_system.get(cid).lifecycle is (
            CommitmentLifecycle.PROPOSED
        )


def test_deadline_crosses_between_subject_review_and_pre_wal(app, monkeypatch):
    """Speculative ACTIVE must roll back if the clock expires before WAL."""
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = _sources(state)
        owner = state._commitment_source_owner
        cid = owner._submit_proposal(_selector(p, x)).result(timeout=30).value
        initial = state.main_loop.export_commitment_state()
        due = state.main_loop._commitment_system.get(cid).deadline.at
        assert due is not None
        clock_reads = 0

        def clock():
            nonlocal clock_reads
            clock_reads += 1
            return (
                due - timedelta(seconds=1)
                if clock_reads == 1
                else (due + timedelta(seconds=1))
            )

        with monkeypatch.context() as patch:
            patch.setattr(commitment_owner_module, "_utcnow", clock)
            with pytest.raises(Exception):
                owner._submit_review(cid).result(timeout=30)
        assert clock_reads >= 2
        assert state.main_loop.export_commitment_state() == initial
        assert owner._pending is None


def test_review_holds_on_current_context_loss(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = _sources(state)
        owner = state._commitment_source_owner
        cid = owner._submit_proposal(_selector(p, x)).result(timeout=30).value
        snapshot = state.main_loop.export_commitment_state()
        context = state.main_loop.context_registry
        get_context = context.get
        with monkeypatch.context() as patch:
            patch.setattr(
                context,
                "get",
                lambda key: replace(get_context(key), status=ContextStatus.SUSPENDED),
            )
            outcome = owner._submit_review(cid).result(timeout=30)
            assert outcome.value == "hold"
        assert state.main_loop.export_commitment_state() == snapshot
        assert owner._pending is None


def test_same_experience_cannot_create_second_responsibility(app):
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = _sources(state)
        owner = state._commitment_source_owner
        cid = owner._submit_proposal(_selector(p, x)).result(timeout=30).value
        before = state.main_loop.export_commitment_state()
        p2 = _chat(state, f"COMMIT_REVIEW_EXPERIENCE:1:{_experience_id(x)}")
        with pytest.raises(AgentRuntimeExecutionError):
            owner._submit_proposal(_selector(p2, x)).result(timeout=30)
        assert state.main_loop.export_commitment_state() == before
        assert state.main_loop._commitment_system.get(cid) is not None


def test_untrusted_caller_or_external_beneficiary_cannot_admit(app):
    with TestClient(app) as client:
        state = client.app.state
        owner = state._commitment_source_owner
        before = state.main_loop.export_commitment_state()
        with pytest.raises(_CommitmentSourceUnavailable):
            owner._submit_proposal("caller proposal or proof")
        with pytest.raises(_CommitmentSourceUnavailable):
            state.main_loop._review_commitment_source(owner, object())
        # No accepted API permits a custom beneficiary or proof object.
        assert state.main_loop.export_commitment_state() == before


def test_pre_wal_source_change_rejects_partial_proposal(app, monkeypatch):
    with TestClient(app) as client:
        state = client.app.state
        x, _, p = _sources(state)
        owner = state._commitment_source_owner
        original = owner._qualify
        calls = 0

        def requalify(selection, event):
            nonlocal calls
            calls += 1
            result = original(selection, event)
            if calls == 2:
                return replace(result, target_record_digest="0" * 64)
            return result

        before = state.main_loop.export_commitment_state()
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_qualify", requalify)
            with pytest.raises(Exception):
                owner._submit_proposal(_selector(p, x)).result(timeout=30)
        assert calls >= 2
        assert state.main_loop.export_commitment_state() == before
        assert owner._pending is None
