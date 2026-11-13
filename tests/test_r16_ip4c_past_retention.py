"""IP-4c (#383 / #381 D1 P1-5, P1-6, D2 P1-3): sources past the retention window.

After compaction archives an event, an owner checking that event in the
Journal uses three stages: retained (checked against the Journal as before),
archived (only with the owner's own committed evidence), otherwise missing.
New sources must be retained, so an archived one is refused as
``SOURCE_PAST_RETENTION`` (or ``SOURCE_MISSING`` when nothing proves it
existed). An admitted Goal continues past the window on its committed
evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from suzka.r15.qualified_proposal import (
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
)
from suzka.r16.outcomes import R16OwnerStatus
from suzka.r16.commitment_owner import _CommitmentSourceSelection
from suzka.r16.goal_owner import _GoalSourceSelection, _GoalSourceUnavailable
from suzka.r16.source_owner import _MotivationSourceRef, _MotivationSourceUnavailable
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from suzka.runtime.journal_compaction import JournalCompactionPolicy
from suzka.runtime.journal_source import (
    JournalSourceReason,
    JournalSourceStage,
    locate_journal_source,
)

from test_r16_u2a_source_owner import _app, settings_in
from test_r16_u2b_goal_owner import chat, preexisting_interest, x_id


@pytest.fixture
def app(tmp_path):
    settings = settings_in(tmp_path / "ip4c")
    settings = settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            ),
            # Small segments so a few turns rotate them.
            "event_journal": settings.event_journal.model_copy(
                update={"max_bytes": 32 * 1024}
            ),
        }
    )
    return _app(settings)


def _stage(state, event_id: str, sequence: int) -> JournalSourceStage:
    stage, _rows = locate_journal_source(
        state.event_journal.inspect(), event_id, sequence
    )
    return stage


def _archive(state, *sources: _MotivationSourceRef) -> None:
    """Chat until every given source event has been archived.

    Compaction runs with the default 30-day window until here; a zero-day
    window then archives every rotated segment at the next clean boundary.
    """
    state.journal_compaction.policy = JournalCompactionPolicy(0)
    for number in range(80):
        if all(
            _stage(state, item.event_id, item.event_sequence)
            is JournalSourceStage.ARCHIVED
            for item in sources
        ):
            return
        chat(state, f"filler turn {number}")
    raise AssertionError(
        f"sources not archived: {state.journal_compaction.last_outcome}"
    )


def _review_event(state, event_type=AgentEventType.GOAL_PROPOSAL_REVIEW):
    """A detached event later than everything retained (for direct reads)."""
    high_water = state.event_journal.inspect().processing_high_water
    return AgentEvent(
        str(uuid4()),
        event_type,
        AgentEventSource.RUNTIME_GOAL_PROPOSAL_REVIEW,
        datetime.now(timezone.utc),
        high_water + 2,
    )


@pytest.mark.parametrize("with_commitment", (False, True))
def test_adopted_goal_continues_after_its_sources_are_archived(
    app, with_commitment: bool
) -> None:
    """A Goal older than the window keeps working: Plan preview succeeds on
    committed evidence after X, U2a, P, the proposal and the adoption (and,
    with a Commitment, its request, proposal and acceptance) are all
    archived (#381 D1 P1-6)."""
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "source experience; fixture")
        preexisting_interest(state, x)
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        owner = state._goal_source_owner
        proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(
            timeout=30
        )
        assert owner._submit_review(proposed.value.subject_id).result(timeout=30).value.status is (
            R16OwnerStatus.ADOPTED
        )
        goal = state.main_loop._goal_system.get(proposed.value.subject_id)
        admission = goal.subject_admission
        late = [admission]
        if with_commitment:
            request = chat(state, f"COMMIT_REVIEW_EXPERIENCE:1:{x_id(x)}")
            commitments = state._commitment_source_owner
            proposal = commitments._submit_proposal(
                _CommitmentSourceSelection(request, x)
            ).result(timeout=30)
            accepted = commitments._submit_review(proposal.value.subject_id).result(timeout=30)
            assert accepted.value.status is R16OwnerStatus.ACCEPTED
            record = state.main_loop._commitment_system.get(proposal.value.subject_id)
            late.append(record.subject_admission)
            _archive(state, request)
        _archive(state, x, p)
        _archive(
            state,
            *(
                _MotivationSourceRef(item.event_id, item.event_sequence, x.episode_id)
                for item in late
            ),
        )
        inspection = state.event_journal.inspect()
        assert inspection.latest_compaction is not None
        assert not any(row.event_id == p.event_id for row in inspection.records)

        preview = state._plan_preview_owner._submit_preview(proposed.value.subject_id).result(
            timeout=30
        )
        assert preview.value.hold is None
        assert preview.value.decision is not None
        assert (preview.value.witness.qualification.commitment_id is not None) is (
            with_commitment
        )


def test_proposed_goal_is_adopted_after_its_sources_are_archived(app) -> None:
    """Adoption is continuation: X, P and the proposal event archived."""
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "source experience; fixture")
        preexisting_interest(state, x)
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        owner = state._goal_source_owner
        proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(
            timeout=30
        )
        genesis = state.main_loop._goal_system.get(proposed.value.subject_id).revision_history[0]
        _archive(
            state,
            x,
            p,
            _MotivationSourceRef(genesis.event_id, genesis.event_sequence, x.episode_id),
        )
        assert owner._submit_review(proposed.value.subject_id).result(timeout=30).value.status is (
            R16OwnerStatus.ADOPTED
        )


def test_archived_request_sequence_probe_is_bounded(app, monkeypatch) -> None:
    """Past the probe bound an archived request is missing (fail closed)."""
    import suzka.r16.source_owner as source_module

    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "source experience; fixture")
        preexisting_interest(state, x)
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        owner = state._goal_source_owner
        proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(
            timeout=30
        )
        genesis = state.main_loop._goal_system.get(proposed.value.subject_id).revision_history[0]
        _archive(state, x, p)
        source = state._motivation_source_owner
        assert source._archived_source_sequence(
            p.event_id, x.event_sequence, genesis.event_sequence
        ) == p.event_sequence
        gap = genesis.event_sequence - p.event_sequence
        monkeypatch.setattr(source_module, "_ARCHIVED_SEQUENCE_PROBE_LIMIT", gap - 1)
        with pytest.raises(_MotivationSourceUnavailable) as refused:
            source._archived_source_sequence(
                p.event_id, x.event_sequence, genesis.event_sequence
            )
        assert refused.value.reason is JournalSourceReason.SOURCE_MISSING
        # The adoption review then holds instead of adopting.
        assert owner._submit_review(proposed.value.subject_id).result(timeout=30).value.status is R16OwnerStatus.HOLD


def test_new_source_past_retention_is_refused_with_its_own_reason(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "old experience; fixture")
        _archive(state, x)
        source_owner = state._motivation_source_owner
        event = _review_event(state)

        with pytest.raises(_MotivationSourceUnavailable) as refused:
            source_owner._read(x, event)
        assert refused.value.reason is JournalSourceReason.SOURCE_PAST_RETENTION

        # Continuation accepts the same source on its committed Experience.
        record, _pending = source_owner._read(x, event, continuation=True)
        assert record.source_event_id == x.event_id

        # A U2a source selection over it is never applied.
        before = state.main_loop.export_motivation_state()
        result = source_owner._submit(x)
        with pytest.raises(Exception):
            result.result(timeout=30)
        assert state.main_loop.export_motivation_state() == before


def test_goal_proposal_over_an_archived_target_is_past_retention(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "old experience; fixture")
        preexisting_interest(state, x)
        _archive(state, x)
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        owner = state._goal_source_owner
        with pytest.raises(_GoalSourceUnavailable) as refused:
            owner._qualify(_GoalSourceSelection(p, x), _review_event(state))
        assert refused.value.reason is JournalSourceReason.SOURCE_PAST_RETENTION


def test_forged_old_sequence_is_missing_not_past_retention(app) -> None:
    """A claimed sequence inside the archived range is not evidence (#364)."""
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "old experience; fixture")
        _archive(state, x)
        source_owner = state._motivation_source_owner
        event = _review_event(state)
        forged = _MotivationSourceRef(str(uuid4()), x.event_sequence, x.episode_id)
        assert _stage(state, forged.event_id, forged.event_sequence) is (
            JournalSourceStage.ARCHIVED
        )
        for continuation in (False, True):
            with pytest.raises(_MotivationSourceUnavailable) as refused:
                source_owner._read(forged, event, continuation=continuation)
            assert refused.value.reason is JournalSourceReason.SOURCE_MISSING
        # The real event id with a wrong archived sequence is missing too.
        shifted = _MotivationSourceRef(x.event_id, x.event_sequence + 1, x.episode_id)
        with pytest.raises(_MotivationSourceUnavailable) as refused:
            source_owner._read(shifted, event, continuation=True)
        assert refused.value.reason is JournalSourceReason.SOURCE_MISSING
        # Beyond the archived range and not retained: missing.
        beyond = _MotivationSourceRef(
            str(uuid4()), event.processing_sequence - 1, x.episode_id
        )
        assert beyond.event_sequence > state.event_journal.inspect().processing_high_water
        assert _stage(state, beyond.event_id, beyond.event_sequence) is (
            JournalSourceStage.MISSING
        )
        with pytest.raises(_MotivationSourceUnavailable) as refused:
            source_owner._read(beyond, event, continuation=True)
        assert refused.value.reason is JournalSourceReason.SOURCE_MISSING


def test_retained_source_is_still_checked_against_the_journal(app) -> None:
    """Inside the window the independent Journal check is kept."""
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "recent experience; fixture")
        assert _stage(state, x.event_id, x.event_sequence) is (
            JournalSourceStage.RETAINED
        )
        source_owner = state._motivation_source_owner
        event = _review_event(state)
        record, _pending = source_owner._read(x, event)
        assert record.source_event_id == x.event_id
        # A retained event read with the wrong sequence fails the Journal
        # check itself, with no retention reason.
        wrong = _MotivationSourceRef(x.event_id, x.event_sequence + 1, x.episode_id)
        with pytest.raises(_MotivationSourceUnavailable) as refused:
            source_owner._read(wrong, event, continuation=True)
        assert refused.value.reason is None


def test_narrative_source_past_retention_is_distinguished(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "old narrative turn; fixture")
        _archive(state, x)
        gate = state.r15_state_port.narrative._source_gate
        with pytest.raises(QualifiedProposalUnavailable) as refused:
            gate._read(QualifiedTurnRef(x.event_id, x.event_sequence, x.episode_id))
        assert refused.value.reason is JournalSourceReason.SOURCE_PAST_RETENTION
        with pytest.raises(QualifiedProposalUnavailable) as refused:
            gate._read(QualifiedTurnRef(str(uuid4()), x.event_sequence, x.episode_id))
        assert refused.value.reason is JournalSourceReason.SOURCE_MISSING
