"""Process-local R14 Attention state-transition behavior tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from suzka.attention.adapters import (
    project_attention_candidates,
    project_commitment,
    project_global_emotion,
    project_goal,
    project_motivation,
    project_working_memory,
)
from suzka.attention.common import (
    ATTENTION_FIXED_POINT_SCALE,
    ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
    ATTENTION_MAX_FOCUS,
    AttentionRevisionReason,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
)
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionCandidateProjection,
    AttentionContinuity,
    AttentionEvent,
    AttentionEventReceipt,
    AttentionSignalVector,
    AttentionSourceWitness,
    AttentionTarget,
)
from suzka.attention.policy import (
    ATTENTION_POLICY_HABITUATION_FOCUSED_DELTA_UNITS,
    ATTENTION_POLICY_HABITUATION_MAX_PENALTY_UNITS,
    ATTENTION_POLICY_HABITUATION_UNFOCUSED_DELTA_UNITS,
    ATTENTION_POLICY_INHIBITION_MAX_PENALTY_UNITS,
    ATTENTION_POLICY_INHIBITION_RECOVERY_UNITS,
    AttentionCandidateDecision,
    AttentionCandidateReason,
    AttentionContinuityBaseline,
    AttentionSignalDimension,
    compete_attention,
    next_habituation_units,
    next_inhibition_units,
    next_streak_counts,
)
from suzka.attention.system import (
    AttentionRefreshResult,
    AttentionSelectedView,
    AttentionSystem,
)
from suzka.context_contracts import ContextRelation
from suzka.emotion_contracts import EmotionState
from suzka.motivation.common import Deadline
from suzka.motivation.commitment import CommitmentLifecycle
from suzka.motivation.goal import GoalLifecycle
from suzka.motivation.motivation import MotivationLifecycle
from suzka.working_memory_contracts import (
    MAX_ITEM_CAPACITY,
    WorkingMemoryDecisionReason,
)


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _event(step: int) -> AttentionEvent:
    return AttentionEvent(
        f"attention:system:{step:04d}",
        10_000 + step,
        NOW + timedelta(seconds=step),
    )


def _policy_projection(
    kind: AttentionTargetKind,
    reference: str,
    event: AttentionEvent,
    *,
    signals: AttentionSignalVector | None = None,
    availability: CandidateAvailability = CandidateAvailability.ELIGIBLE,
    source_revision: int = 2,
    source_digest: str = "a" * 64,
    source_event: AttentionEvent | None = None,
    rendered_bytes: int = 64,
) -> AttentionCandidateProjection:
    """Create policy-only inputs without inventing any source-domain payload."""

    source_kind = {
        AttentionTargetKind.WORKING_MEMORY: AttentionSourceKind.WORKING_MEMORY,
        AttentionTargetKind.MOTIVATION: AttentionSourceKind.MOTIVATION,
        AttentionTargetKind.GOAL: AttentionSourceKind.GOAL,
        AttentionTargetKind.COMMITMENT: AttentionSourceKind.COMMITMENT,
    }[kind]
    target = AttentionTarget(kind, reference)
    source = AttentionSourceWitness(
        kind=source_kind,
        reference=reference,
        revision=source_revision,
        digest=source_digest,
        digest_kind=(
            SourceDigestKind.ATTENTION_PROJECTION
            if kind is AttentionTargetKind.WORKING_MEMORY
            else SourceDigestKind.UPSTREAM_AUTHORITY
        ),
        target_kind=kind,
        target_reference=reference,
        source_event_id=None if source_event is None else source_event.event_id,
        source_event_sequence=None
        if source_event is None
        else source_event.event_sequence,
        source_occurred_at=None
        if source_event is None
        else source_event.occurred_at,
    )
    eligible = availability is CandidateAvailability.ELIGIBLE
    return AttentionCandidateProjection._create(
        target=target,
        source=source,
        signals=signals or AttentionSignalVector(),
        event=event,
        availability=availability,
        rendered_bytes=rendered_bytes if eligible else None,
        rendered_digest=target.candidate_id if eligible else None,
    )


def _goal_projection(
    event: AttentionEvent,
    reference: str = "1" * 64,
    *,
    signals: AttentionSignalVector | None = None,
    availability: CandidateAvailability = CandidateAvailability.ELIGIBLE,
    source_revision: int = 2,
    source_digest: str = "a" * 64,
    source_event: AttentionEvent | None = None,
    rendered_bytes: int = 64,
) -> AttentionCandidateProjection:
    return _policy_projection(
        AttentionTargetKind.GOAL,
        reference,
        event,
        signals=signals,
        availability=availability,
        source_revision=source_revision,
        source_digest=source_digest,
        source_event=source_event,
        rendered_bytes=rendered_bytes,
    )


def _decision_by_id(
    result: AttentionRefreshResult, candidate_id: str
) -> AttentionCandidateDecision:
    competition = result.view.competition
    assert competition is not None
    return next(
        decision
        for decision in competition.decisions
        if decision.candidate_id == candidate_id
    )


def _candidate_by_id(
    snapshot: AttentionContinuity, candidate_id: str
) -> AttentionCandidateContinuity:
    return next(
        candidate
        for candidate in snapshot.candidates
        if candidate.candidate_id == candidate_id
    )


def _view_target_ids(view: AttentionSelectedView, name: str) -> tuple[str, ...]:
    return tuple(target.candidate_id for target in getattr(view, name))


def test_bootstrap_empty_refresh_and_current_exact_retry_are_receipted() -> None:
    system = AttentionSystem()
    bootstrap = system.snapshot()
    bootstrap_view = system.selected_view()
    event = _event(1)

    assert bootstrap == AttentionContinuity.bootstrap()
    assert bootstrap_view.revision == 0
    assert bootstrap_view.event is None
    assert bootstrap_view.focused_targets == ()
    assert bootstrap_view.unfinished_targets == ()
    assert bootstrap_view.competition is None
    assert bootstrap_view.prompt is None

    first = system.refresh((), event)

    assert type(first) is AttentionRefreshResult
    assert type(first.view) is AttentionSelectedView
    assert first.replayed is False
    assert first.snapshot.revision == 1
    assert first.snapshot.last_event == event
    assert first.snapshot.candidates == ()
    assert first.snapshot.focused_ids == ()
    assert first.snapshot.unfinished_ids == ()
    assert len(first.snapshot.revision_history) == 1
    assert first.snapshot.revision_history[0].reason is AttentionRevisionReason.STATE_UPDATE
    assert len(first.snapshot.receipts) == 1
    assert type(first.receipt) is AttentionEventReceipt
    assert first.receipt == first.snapshot.receipts[-1]
    assert first.receipt.event == event
    assert first.view.revision == 1
    assert first.view.event == event
    assert first.view.state_digest == first.snapshot.state_digest
    assert first.view.authority_digest == first.snapshot.authority_digest
    assert first.view.focused_targets == ()
    assert first.view.unfinished_targets == ()
    assert first.view.competition is not None
    assert first.view.prompt is not None
    assert first.view.competition.focused_ids == ()

    committed_bytes = first.snapshot.canonical_bytes()
    replay = system.refresh((), event)

    assert replay.replayed is True
    assert replay.snapshot.canonical_bytes() == committed_bytes
    assert replay.receipt == first.receipt
    assert replay.view == first.view
    assert system.snapshot().canonical_bytes() == committed_bytes
    assert len(system.snapshot().revision_history) == 1
    assert len(system.snapshot().receipts) == 1


def test_request_receipt_digest_is_separate_from_prior_bound_competition_digest() -> None:
    empty_history = AttentionSystem()
    occupied_history = AttentionSystem()
    first_event = _event(1)
    candidate = _goal_projection(first_event)
    empty_history.refresh((), first_event)
    occupied_history.refresh((candidate,), first_event)

    current_event = _event(2)
    current_candidate = _goal_projection(current_event)
    empty_result = empty_history.refresh((current_candidate,), current_event)
    occupied_result = occupied_history.refresh((current_candidate,), current_event)

    assert empty_result.receipt.input_digest == occupied_result.receipt.input_digest
    assert empty_result.view.competition is not None
    assert occupied_result.view.competition is not None
    assert (
        empty_result.view.competition.prior_authority_digest
        != occupied_result.view.competition.prior_authority_digest
    )
    assert (
        empty_result.view.competition.input_digest
        != occupied_result.view.competition.input_digest
    )
    assert empty_result.receipt.input_digest != empty_result.view.competition.input_digest
    assert occupied_result.receipt.input_digest != occupied_result.view.competition.input_digest


def test_real_source_adapters_feed_exact_current_targets_without_mutating_sources() -> None:
    from test_attention_adapters import _item, _view
    from test_r13_projection import (
        _make_commitment_system,
        _make_goal_system,
        _make_motivation_system,
    )

    event = _event(1)
    motivation_system = _make_motivation_system((MotivationLifecycle.ACTIVE,))
    goal_system = _make_goal_system((GoalLifecycle.ADOPTED,))
    commitment_system = _make_commitment_system((CommitmentLifecycle.ACTIVE,))
    motivation_snapshot = motivation_system.snapshot()
    goal_snapshot = goal_system.snapshot()
    commitment_snapshot = commitment_system.snapshot()
    source_signatures = tuple(
        (
            snapshot.authority_digest,
            snapshot.serialized_bytes,
            tuple(record.record_digest for record in snapshot.records),
        )
        for snapshot in (motivation_snapshot, goal_snapshot, commitment_snapshot)
    )

    item = _item("attention-exact-wm-view")
    wm_view = _view(
        item,
        content="resolved row from the exact R08 view",
        reason=WorkingMemoryDecisionReason.SELECTED,
        relation=ContextRelation.SAME_CONTEXT,
        compatibility=1.0,
        effective_score=0.6 * item.activation + 0.4 * item.salience,
    )
    projections = (
        project_motivation(motivation_snapshot.records[0], event=event),
        project_goal(goal_snapshot.records[0], event=event),
        project_commitment(commitment_snapshot.records[0], event=event),
        project_working_memory(item, revision=1, event=event, view=wm_view),
    )

    for projection in projections:
        system = AttentionSystem()
        result = system.refresh((projection,), event)
        stored = result.snapshot.candidates[0]
        assert stored.target == projection.target
        assert stored.source == projection.source
        assert stored.availability is projection.availability
        assert result.view.competition is not None
        assert result.view.competition.decisions[0].candidate_id == projection.candidate_id
        assert _view_target_ids(result.view, "focused_targets") == (
            projection.candidate_id,
        )
        assert "goal description 000" not in repr(result.snapshot)
        assert "private subject 000" not in repr(result.snapshot)

    assert source_signatures == tuple(
        (
            snapshot.authority_digest,
            snapshot.serialized_bytes,
            tuple(record.record_digest for record in snapshot.records),
        )
        for snapshot in (motivation_system.snapshot(), goal_system.snapshot(), commitment_system.snapshot())
    )
    assert item.activation == pytest.approx(0.8)
    assert item.salience == pytest.approx(0.6)
    assert "resolved row from the exact R08 view" not in repr(system.snapshot())


def test_complete_4192_candidate_universe_is_retained_not_only_selected_rows() -> None:
    from test_attention_adapters import _item
    from test_r13_projection import (
        _commitment_proposal,
        _goal_proposal,
        _make_motivation_system,
    )
    from suzka.motivation.commitment import (
        CommitmentAdmissionReason,
        CommitmentSubjectAdmission,
    )
    from suzka.motivation.commitment_system import (
        CommitmentMutationEvidence,
        CommitmentSystem,
    )
    from suzka.motivation.goal import GoalAdmissionReason, GoalSubjectAdmission
    from suzka.motivation.goal_system import GoalMutationEvidence, GoalSystem

    event = _event(1)
    items = tuple(
        _item(
            f"attention-full-universe-{index:04d}",
            activation=0.4,
            salience=0.2,
        )
        for index in range(MAX_ITEM_CAPACITY)
    )
    motivations = _make_motivation_system((MotivationLifecycle.ACTIVE,) * 32).snapshot()

    def source_time(sequence: int) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=sequence)

    goal_system = GoalSystem()
    goal_proposals = tuple(_goal_proposal(index) for index in range(32))
    for index, proposal in enumerate(goal_proposals):
        sequence = 100 + index
        goal_system.ingest_proposal(
            proposal,
            GoalMutationEvidence(
                f"attention:test:goal:ingest:{index:03d}",
                sequence,
                source_time(sequence),
                proposal.evidence_refs,
            ),
        )
    for index, proposal in enumerate(goal_proposals):
        sequence = 200 + index
        source_event = GoalMutationEvidence(
            f"attention:test:goal:adopt:{index:03d}",
            sequence,
            source_time(sequence),
            proposal.evidence_refs,
        )
        admission = GoalSubjectAdmission(
            goal_id=proposal.goal_id,
            proposal_digest=proposal.proposal_digest,
            evidence_refs=tuple(sorted(item.reference for item in proposal.evidence_refs)),
            event_id=source_event.event_id,
            event_sequence=source_event.event_sequence,
            reason=GoalAdmissionReason.SUBJECT_ENDORSEMENT,
        )
        goal_system.adopt(proposal.goal_id, admission, source_event)
    goals = goal_system.snapshot()

    commitment_system = CommitmentSystem()
    commitment_proposals = tuple(_commitment_proposal(index) for index in range(32))
    for index, proposal in enumerate(commitment_proposals):
        sequence = 300 + index
        commitment_system.ingest_proposal(
            proposal,
            CommitmentMutationEvidence(
                f"attention:test:commitment:ingest:{index:03d}",
                sequence,
                source_time(sequence),
                proposal.evidence_refs,
            ),
        )
    for index, proposal in enumerate(commitment_proposals):
        sequence = 400 + index
        source_event = CommitmentMutationEvidence(
            f"attention:test:commitment:accept:{index:03d}",
            sequence,
            source_time(sequence),
            proposal.evidence_refs,
        )
        admission = CommitmentSubjectAdmission(
            commitment_id=proposal.commitment_id,
            proposal_digest=proposal.proposal_digest,
            beneficiary=proposal.beneficiary,
            scope=proposal.scope,
            deadline=proposal.deadline,
            evidence_refs=tuple(sorted(item.reference for item in proposal.evidence_refs)),
            event_id=source_event.event_id,
            event_sequence=source_event.event_sequence,
            reason=CommitmentAdmissionReason.SUBJECT_ENDORSEMENT,
        )
        commitment_system.accept(proposal.commitment_id, admission, source_event)
    commitments = commitment_system.snapshot()

    source_snapshots = (motivations, goals, commitments)
    source_signatures = tuple(
        (
            snapshot.authority_digest,
            snapshot.serialized_bytes,
            tuple(record.record_digest for record in snapshot.records),
        )
        for snapshot in source_snapshots
    )
    projections = project_attention_candidates(
        working_memory_items=items,
        working_memory_revision=1,
        working_memory_view=None,
        motivation_snapshot=motivations,
        goal_snapshot=goals,
        commitment_snapshot=commitments,
        event=event,
    )

    assert len(projections) == 4_192
    assert sum(
        item.target.kind is AttentionTargetKind.WORKING_MEMORY
        and item.availability is CandidateAvailability.UNAVAILABLE
        for item in projections
    ) == 4_096
    for kind in (
        AttentionTargetKind.MOTIVATION,
        AttentionTargetKind.GOAL,
        AttentionTargetKind.COMMITMENT,
    ):
        assert sum(item.target.kind is kind for item in projections) == 32
    source_records = {
        AttentionTargetKind.MOTIVATION: {
            record.motivation_id: record for record in motivations.records
        },
        AttentionTargetKind.GOAL: {record.goal_id: record for record in goals.records},
        AttentionTargetKind.COMMITMENT: {
            record.commitment_id: record for record in commitments.records
        },
    }
    for projection in projections:
        source = projection.source
        if projection.target.kind is AttentionTargetKind.WORKING_MEMORY:
            assert source.source_event_id is None
            assert source.source_event_sequence is None
            assert source.source_occurred_at is None
            continue
        record = source_records[projection.target.kind][projection.target.reference]
        latest = record.revision_history[-1]
        assert source.source_event_id == latest.event_id
        assert source.source_event_sequence == latest.event_sequence
        assert source.source_occurred_at == latest.created_at

    result = AttentionSystem().refresh(projections, event)

    assert len(result.snapshot.candidates) == len(projections) == 4_192
    assert len(result.view.competition.decisions) == 4_192  # type: ignore[union-attr]
    stored_by_id = {item.candidate_id: item for item in result.snapshot.candidates}
    assert set(stored_by_id) == {item.candidate_id for item in projections}
    for projection in projections:
        assert stored_by_id[projection.candidate_id].source == projection.source
    assert sum(
        item.target.kind is AttentionTargetKind.WORKING_MEMORY
        and item.availability is CandidateAvailability.UNAVAILABLE
        for item in result.snapshot.candidates
    ) == 4_096
    assert len(result.snapshot.focused_ids) <= ATTENTION_MAX_FOCUS
    assert len(result.snapshot.candidates) > len(result.snapshot.focused_ids)
    assert source_signatures == tuple(
        (
            snapshot.authority_digest,
            snapshot.serialized_bytes,
            tuple(record.record_digest for record in snapshot.records),
        )
        for snapshot in source_snapshots
    )


def test_tied_scores_and_input_permutations_produce_identical_state_and_views() -> None:
    event = _event(1)
    projections = (
        _policy_projection(AttentionTargetKind.GOAL, "1" * 64, event),
        _policy_projection(AttentionTargetKind.MOTIVATION, "2" * 64, event),
        _policy_projection(AttentionTargetKind.COMMITMENT, "3" * 64, event),
    )
    first = AttentionSystem().refresh(projections, event)
    permuted = AttentionSystem().refresh(tuple(reversed(projections)), event)

    assert first.view.competition is not None
    assert permuted.view.competition is not None
    expected_order = tuple(sorted(item.candidate_id for item in projections))
    assert first.view.competition.focus_order == expected_order
    assert permuted.view.competition.focus_order == expected_order
    assert first.view.competition.input_digest == permuted.view.competition.input_digest
    assert first.view.competition.result_digest == permuted.view.competition.result_digest
    assert first.snapshot.state_digest == permuted.snapshot.state_digest
    assert first.snapshot.authority_digest == permuted.snapshot.authority_digest
    assert first.snapshot.canonical_bytes() == permuted.snapshot.canonical_bytes()
    assert first.view == permuted.view


def test_high_global_arousal_only_caps_focus_and_does_not_change_candidate_scores() -> None:
    event = _event(1)
    projections = tuple(
        _policy_projection(AttentionTargetKind.GOAL, f"{index:064x}", event)
        for index in range(20)
    )
    high_arousal = project_global_emotion(EmotionState(arousal=0.75), event=event)

    ordinary = AttentionSystem().refresh(projections, event)
    constrained = AttentionSystem().refresh(
        projections, event, global_emotion=high_arousal
    )

    assert ordinary.view.competition is not None
    assert constrained.view.competition is not None
    assert ordinary.view.competition.focus_capacity == ATTENTION_MAX_FOCUS
    assert constrained.view.competition.focus_capacity == ATTENTION_HIGH_AROUSAL_MAX_FOCUS
    assert len(ordinary.snapshot.focused_ids) == ATTENTION_MAX_FOCUS
    assert len(constrained.snapshot.focused_ids) == ATTENTION_HIGH_AROUSAL_MAX_FOCUS
    assert tuple(
        (item.candidate_id, item.score_units)
        for item in ordinary.view.competition.decisions
    ) == tuple(
        (item.candidate_id, item.score_units)
        for item in constrained.view.competition.decisions
    )


def test_source_kind_never_overrides_fixed_numeric_competition() -> None:
    event = _event(1)
    low_goal = _policy_projection(
        AttentionTargetKind.GOAL,
        "1" * 64,
        event,
        signals=AttentionSignalVector(urgency=0.0),
    )
    strong_motivation = _policy_projection(
        AttentionTargetKind.MOTIVATION,
        "2" * 64,
        event,
        signals=AttentionSignalVector(strength=1.0, persistence=1.0),
    )

    result = AttentionSystem().refresh((low_goal, strong_motivation), event)

    assert result.view.competition is not None
    scores = {item.candidate_id: item.score_units for item in result.view.competition.decisions}
    assert result.view.competition.focus_order[0] == strong_motivation.candidate_id
    assert scores[strong_motivation.candidate_id] > scores[low_goal.candidate_id]
    assert result.view.competition.decisions[
        next(
            index
            for index, item in enumerate(result.view.competition.decisions)
            if item.candidate_id == strong_motivation.candidate_id
        )
    ].reason is AttentionCandidateReason.SELECTED_FOCUS


def test_unknown_goal_urgency_is_neutral_but_absent_and_far_deadlines_are_measured_zero() -> None:
    from test_r13_projection import _make_goal_system

    event = _event(1)
    no_deadline_snapshot = _make_goal_system(
        (GoalLifecycle.ADOPTED,), deadline=Deadline.without_deadline()
    ).snapshot()
    distant_snapshot = _make_goal_system(
        (GoalLifecycle.ADOPTED,), deadline=Deadline(NOW + timedelta(hours=48))
    ).snapshot()
    no_deadline = project_goal(no_deadline_snapshot.records[0], event=event)
    distant = project_goal(distant_snapshot.records[0], event=event)
    generic_unknown = _policy_projection(
        AttentionTargetKind.GOAL,
        no_deadline.target.reference,
        event,
        signals=AttentionSignalVector(),
        source_revision=no_deadline.source.revision,
        source_digest=no_deadline.source.digest,
        rendered_bytes=no_deadline.rendered_bytes or 64,
    )
    measured_no_deadline = AttentionSystem().refresh((no_deadline,), event)
    measured_distant = AttentionSystem().refresh((distant,), event)
    unknown = AttentionSystem().refresh((generic_unknown,), event)

    assert no_deadline.signals.urgency == distant.signals.urgency == 0.0
    assert no_deadline.source.revision == distant.source.revision
    assert no_deadline.source.digest != distant.source.digest
    assert measured_no_deadline.view.competition is not None
    assert measured_distant.view.competition is not None
    assert unknown.view.competition is not None
    assert measured_no_deadline.view.competition.decisions[0].base_score_units == 400_000
    assert measured_distant.view.competition.decisions[0].base_score_units == 400_000
    assert unknown.view.competition.decisions[0].base_score_units == 500_000
    assert (
        AttentionSignalDimension.URGENCY
        not in measured_no_deadline.view.competition.decisions[0].missing_dimensions
    )
    assert (
        AttentionSignalDimension.URGENCY
        in unknown.view.competition.decisions[0].missing_dimensions
    )


def test_event_time_can_change_projection_urgency_without_changing_primary_source() -> None:
    from test_r13_projection import _make_goal_system

    source = _make_goal_system(
        (GoalLifecycle.ADOPTED,),
        deadline=Deadline(NOW + timedelta(hours=2)),
    ).snapshot()
    record = source.records[0]
    first_event = _event(1)
    second_event = _event(2)
    first_projection = project_goal(record, event=first_event)
    second_projection = project_goal(record, event=second_event)
    before = (record.record_digest, record.revision, source.authority_digest)
    system = AttentionSystem()

    first = system.refresh((first_projection,), first_event)
    second = system.refresh((second_projection,), second_event)

    assert first_projection.source.digest == second_projection.source.digest
    assert first_projection.source.revision == second_projection.source.revision
    assert first_projection.projection_digest != second_projection.projection_digest
    assert first_projection.signals.urgency != second_projection.signals.urgency
    assert first.view.competition is not None
    assert second.view.competition is not None
    assert first.view.competition.input_digest != second.view.competition.input_digest
    assert second.snapshot.revision == first.snapshot.revision + 1
    first_state_candidate = first.snapshot.candidates[0]
    second_state_candidate = second.snapshot.candidates[0]
    assert first_state_candidate.source == second_state_candidate.source
    assert before == (record.record_digest, record.revision, source.authority_digest)


def test_context_view_change_updates_projection_not_working_memory_primary_witness() -> None:
    from test_attention_adapters import _item, _view

    item = _item("attention-dynamic-context")
    first_event = _event(1)
    second_event = _event(2)
    same_context = _view(
        item,
        content="same rendered memory row",
        reason=WorkingMemoryDecisionReason.SELECTED,
        relation=ContextRelation.SAME_CONTEXT,
        compatibility=1.0,
        effective_score=0.6 * item.activation + 0.4 * item.salience,
    )
    unrelated_context = _view(
        item,
        content="same rendered memory row",
        reason=WorkingMemoryDecisionReason.SELECTED,
        relation=ContextRelation.UNRELATED,
        compatibility=0.2,
        effective_score=(0.6 * item.activation + 0.4 * item.salience) * 0.2,
    )
    first_projection = project_working_memory(
        item, revision=1, event=first_event, view=same_context
    )
    second_projection = project_working_memory(
        item, revision=1, event=second_event, view=unrelated_context
    )
    system = AttentionSystem()

    first = system.refresh((first_projection,), first_event)
    second = system.refresh((second_projection,), second_event)

    assert first_projection.source.revision == second_projection.source.revision == 1
    assert first_projection.source.digest == second_projection.source.digest
    assert first_projection.projection_digest != second_projection.projection_digest
    assert first_projection.signals.context_compatibility == 1.0
    assert second_projection.signals.context_compatibility == 0.2
    assert first.snapshot.candidates[0].source == second.snapshot.candidates[0].source
    assert second.snapshot.revision == first.snapshot.revision + 1


def test_competition_uses_prior_counters_before_advancing_each_fresh_event_once() -> None:
    from test_attention_policy import _prior

    prior_event = _event(0)
    event = _event(1)
    prior_projections = tuple(
        _policy_projection(AttentionTargetKind.GOAL, f"{index:064x}", prior_event)
        for index in range(9)
    )
    focused_id = prior_projections[0].candidate_id
    streaks = {
        projection.candidate_id: (
            100_000 + index * 10_000,
            250_000 + index * 10_000,
            2 if projection.candidate_id == focused_id else 0,
            0 if projection.candidate_id == focused_id else 3,
        )
        for index, projection in enumerate(prior_projections)
    }
    prior = _prior(
        prior_projections,
        event=prior_event,
        focused_ids=(focused_id,),
        streaks=streaks,
    )
    current = tuple(
        _policy_projection(AttentionTargetKind.GOAL, f"{index:064x}", event)
        for index in range(9)
    )
    emotion = project_global_emotion(EmotionState(arousal=0.75), event=event)
    expected = compete_attention(current, prior, event, emotion)
    result = AttentionSystem(prior).refresh(current, event, global_emotion=emotion)

    assert result.view.competition == expected
    assert result.view.competition is not None
    assert len(result.snapshot.focused_ids) == ATTENTION_HIGH_AROUSAL_MAX_FOCUS
    for previous in prior.candidates:
        focused = previous.candidate_id in expected.focused_ids
        current_candidate = _candidate_by_id(result.snapshot, previous.candidate_id)
        decision = _decision_by_id(result, previous.candidate_id)
        expected_habituation = (
            min(
                ATTENTION_FIXED_POINT_SCALE,
                previous.habituation + ATTENTION_POLICY_HABITUATION_FOCUSED_DELTA_UNITS,
            )
            if focused
            else max(
                0,
                previous.habituation - ATTENTION_POLICY_HABITUATION_UNFOCUSED_DELTA_UNITS,
            )
        )
        assert current_candidate.habituation == next_habituation_units(
            previous.habituation, focused=focused
        )
        assert current_candidate.habituation == expected_habituation
        assert current_candidate.inhibition == next_inhibition_units(previous.inhibition)
        assert current_candidate.inhibition == max(
            0, previous.inhibition - ATTENTION_POLICY_INHIBITION_RECOVERY_UNITS
        )
        assert (current_candidate.focused_event_count, current_candidate.unattended_event_count) == (
            next_streak_counts(
                previous.focused_event_count,
                previous.unattended_event_count,
                focused=focused,
            )
        )
        assert decision.habituation_penalty_units == (
            previous.habituation
            * ATTENTION_POLICY_HABITUATION_MAX_PENALTY_UNITS
            // ATTENTION_FIXED_POINT_SCALE
        )
        assert decision.inhibition_penalty_units == (
            previous.inhibition
            * ATTENTION_POLICY_INHIBITION_MAX_PENALTY_UNITS
            // ATTENTION_FIXED_POINT_SCALE
        )
    assert result.snapshot.revision == prior.revision + 1
    assert len(result.snapshot.revision_history) == len(prior.revision_history) + 1
    assert len(result.snapshot.receipts) == len(prior.receipts) + 1


def test_salient_challenger_eventually_switches_after_repeat_focus_habituation() -> None:
    from test_attention_policy import _prior

    prior_event = _event(0)
    incumbent_references = tuple(f"wm-{index:064x}" for index in range(1, 17))
    incumbent_before = tuple(
        _policy_projection(AttentionTargetKind.WORKING_MEMORY, reference, prior_event)
        for reference in incumbent_references
    )
    incumbent_ids = tuple(item.candidate_id for item in incumbent_before)
    challenger_reference = f"wm-{17:064x}"
    challenger_id = _policy_projection(
        AttentionTargetKind.WORKING_MEMORY,
        challenger_reference,
        prior_event,
    ).candidate_id
    prior = _prior(
        incumbent_before,
        event=prior_event,
        focused_ids=incumbent_ids,
        streaks={candidate_id: (500_000, 0, 1, 0) for candidate_id in incumbent_ids},
    )
    system = AttentionSystem(prior)
    switched_at: int | None = None
    first_result = None
    final_result = None

    for step in range(1, 7):
        event = _event(step)
        challenger_signals = (
            AttentionSignalVector(activation=0.5, salience=1.0, context_compatibility=0.0)
            if step == 1
            else AttentionSignalVector(activation=0.5, salience=0.5, context_compatibility=1.0)
        )
        current = tuple(
            _policy_projection(
                AttentionTargetKind.WORKING_MEMORY,
                reference,
                event,
            )
            for reference in incumbent_references
        ) + (
            _policy_projection(
                AttentionTargetKind.WORKING_MEMORY,
                challenger_reference,
                event,
                signals=challenger_signals,
            ),
        )
        before = system.snapshot()
        expected = compete_attention(current, before, event)
        result = system.refresh(current, event)
        if first_result is None:
            first_result = result
        final_result = result
        assert result.view.competition == expected
        assert result.view.competition is not None
        if step == 1:
            assert set(result.snapshot.focused_ids) == set(incumbent_ids)
            assert challenger_id not in result.snapshot.focused_ids
            challenger_decision = _decision_by_id(result, challenger_id)
            incumbent_decision = _decision_by_id(result, incumbent_ids[0])
            assert challenger_decision.base_score_units == 550_000
            assert challenger_decision.switch_cost_units == 50_000
            assert challenger_decision.score_units == 500_000
            assert incumbent_decision.base_score_units == 500_000
            assert incumbent_decision.focus_bonus_units == 112_500
            assert incumbent_decision.habituation_penalty_units == 75_000
            assert incumbent_decision.score_units == 537_500
            assert result.view.competition.focus_order == tuple(sorted(incumbent_ids))
        if set(result.snapshot.focused_ids) != set(incumbent_ids):
            switched_at = step
            assert challenger_id in result.snapshot.focused_ids
            assert challenger_id in result.view.competition.focused_ids
            assert len(result.snapshot.focused_ids) == ATTENTION_MAX_FOCUS
            selected_incumbents = set(result.snapshot.focused_ids).intersection(incumbent_ids)
            displaced_incumbents = set(incumbent_ids).difference(selected_incumbents)
            assert len(selected_incumbents) == ATTENTION_MAX_FOCUS - 1
            assert len(displaced_incumbents) == 1
            assert displaced_incumbents.issubset(result.snapshot.unfinished_ids)
            break

    assert switched_at == 5
    assert first_result is not None
    assert final_result is not None
    assert set(first_result.snapshot.focused_ids) == set(incumbent_ids)
    assert all(
        _candidate_by_id(first_result.snapshot, candidate_id).habituation == 550_000
        for candidate_id in incumbent_ids
    )
    assert final_result.view.competition is not None
    final_challenger = _decision_by_id(final_result, challenger_id)
    assert final_challenger.score_units == 550_000


@pytest.mark.parametrize(
    "status",
    (CandidateAvailability.UNAVAILABLE, CandidateAvailability.INACTIVE),
)
def test_unavailable_or_inactive_rows_remain_but_lose_references_and_no_cache(
    status: CandidateAvailability,
) -> None:
    reference = "1" * 64
    first_event = _event(1)
    first_projection = _goal_projection(first_event, reference)
    system = AttentionSystem()
    first = system.refresh((first_projection,), first_event)

    status_event = _event(2)
    status_projection = _policy_projection(
        AttentionTargetKind.GOAL,
        reference,
        status_event,
        availability=status,
    )
    unavailable = system.refresh((status_projection,), status_event)
    status_candidate = _candidate_by_id(unavailable.snapshot, first_projection.candidate_id)

    assert status_candidate.availability is status
    assert first_projection.candidate_id not in unavailable.snapshot.focused_ids
    assert first_projection.candidate_id not in unavailable.snapshot.unfinished_ids
    assert first_projection.candidate_id not in _view_target_ids(
        unavailable.view, "focused_targets"
    )
    assert first_projection.candidate_id not in _view_target_ids(
        unavailable.view, "unfinished_targets"
    )

    absent = system.refresh((), _event(3))
    assert absent.snapshot.candidates == ()
    assert absent.snapshot.focused_ids == ()
    assert absent.snapshot.unfinished_ids == ()

    reintroduced = system.refresh((_goal_projection(_event(4), reference),), _event(4))
    decision = _decision_by_id(reintroduced, first_projection.candidate_id)
    assert decision.continuity_baseline is AttentionContinuityBaseline.NEW
    assert not decision.prior_candidate_present
    assert _candidate_by_id(reintroduced.snapshot, first_projection.candidate_id).habituation == 50_000
    assert first.snapshot.candidates[0].availability is CandidateAvailability.ELIGIBLE


def test_unfinished_references_survive_only_while_the_complete_current_set_has_them() -> None:
    from test_attention_policy import _prior

    prior_event = _event(0)
    old_unfinished = _policy_projection(
        AttentionTargetKind.GOAL, "1" * 64, prior_event
    )
    challengers = tuple(
        _policy_projection(
            AttentionTargetKind.WORKING_MEMORY,
            f"wm-{index:064x}",
            prior_event,
            signals=AttentionSignalVector(
                activation=1.0,
                salience=1.0,
                context_compatibility=1.0,
            ),
        )
        for index in range(1, 9)
    )
    prior = _prior(
        (old_unfinished, *challengers),
        event=prior_event,
        unfinished_ids=(old_unfinished.candidate_id,),
        streaks={old_unfinished.candidate_id: (0, 0, 0, 0)},
    )
    system = AttentionSystem(prior)
    event = _event(1)
    current = (
        _goal_projection(event, old_unfinished.target.reference),
        *(
            _policy_projection(
                AttentionTargetKind.WORKING_MEMORY,
                projection.target.reference,
                event,
                signals=projection.signals,
            )
            for projection in challengers
        ),
    )
    emotion = project_global_emotion(EmotionState(arousal=0.75), event=event)
    retained = system.refresh(current, event, global_emotion=emotion)

    assert old_unfinished.candidate_id in retained.snapshot.unfinished_ids
    assert old_unfinished.candidate_id in {
        candidate.candidate_id for candidate in retained.snapshot.candidates
    }

    without_old = system.refresh(
        tuple(
            _policy_projection(
                projection.target.kind,
                projection.target.reference,
                _event(2),
                signals=projection.signals,
            )
            for projection in current[1:]
        ),
        _event(2),
        global_emotion=project_global_emotion(EmotionState(arousal=0.75), event=_event(2)),
    )

    assert old_unfinished.candidate_id not in without_old.snapshot.unfinished_ids
    assert old_unfinished.candidate_id not in without_old.snapshot.focused_ids
    assert old_unfinished.candidate_id not in {
        candidate.candidate_id for candidate in without_old.snapshot.candidates
    }


def test_same_primary_revision_must_keep_its_digest_and_source_revision_is_independent() -> None:
    first_event = _event(1)
    event2 = _event(2)
    original = _goal_projection(first_event, source_revision=5, source_digest="a" * 64)
    system = AttentionSystem()
    first = system.refresh((original,), first_event)
    assert first.snapshot.revision == 1
    assert first.snapshot.candidates[0].source.revision == 5

    changed_digest = _goal_projection(
        event2, source_revision=5, source_digest="b" * 64
    )
    before_rejection = system.snapshot().canonical_bytes()
    with pytest.raises((TypeError, ValueError)):
        system.refresh((changed_digest,), event2)
    assert system.snapshot().canonical_bytes() == before_rejection

    regressed = _goal_projection(event2, source_revision=4, source_digest="a" * 64)
    with pytest.raises((TypeError, ValueError)):
        system.refresh((regressed,), event2)
    assert system.snapshot().canonical_bytes() == before_rejection

    unchanged_primary = _goal_projection(
        event2, source_revision=5, source_digest="a" * 64
    )
    second = system.refresh((unchanged_primary,), event2)
    assert second.snapshot.revision == 2
    assert second.snapshot.candidates[0].source == first.snapshot.candidates[0].source
    assert len(second.snapshot.revision_history) == 2
    assert len(second.snapshot.receipts) == 2


def test_same_primary_revision_rejects_changed_upstream_event_identity() -> None:
    first_event = _event(1)
    second_event = _event(2)
    upstream_event = AttentionEvent(
        "upstream:goal:revision-one", 90, NOW - timedelta(minutes=1)
    )
    first_projection = _goal_projection(
        first_event,
        source_revision=5,
        source_digest="a" * 64,
        source_event=upstream_event,
    )
    system = AttentionSystem()
    accepted = system.refresh((first_projection,), first_event)
    before = accepted.snapshot.canonical_bytes()
    changed_source_event = AttentionEvent(
        "upstream:goal:revision-one-altered", 91, NOW
    )
    conflicting = _goal_projection(
        second_event,
        source_revision=5,
        source_digest="a" * 64,
        source_event=changed_source_event,
    )

    with pytest.raises((TypeError, ValueError)):
        system.refresh((conflicting,), second_event)

    assert system.snapshot().canonical_bytes() == before


def test_attention_revision_one_accepts_source_revision_zero_from_a_real_adapter() -> None:
    from test_r13_projection import _make_motivation_system

    source = _make_motivation_system((MotivationLifecycle.ACTIVE,)).snapshot()
    record = source.records[0]
    assert record.revision == 0
    system = AttentionSystem()

    first_event = _event(1)
    first_projection = project_motivation(record, event=first_event)
    first = system.refresh((first_projection,), first_event)
    second_event = _event(2)
    second_projection = project_motivation(record, event=second_event)
    second = system.refresh((second_projection,), second_event)

    assert first_projection.source == second_projection.source
    assert first_projection.projection_digest != second_projection.projection_digest
    assert first.snapshot.revision == 1
    assert second.snapshot.revision == 2
    assert first.snapshot.candidates[0].source.revision == 0
    assert second.snapshot.candidates[0].source.revision == 0


def test_input_conflicts_for_a_committed_event_are_rejected_without_state_change() -> None:
    from suzka.attention.adapters import GlobalEmotionProjection

    event = _event(1)
    original = _goal_projection(
        event,
        signals=AttentionSignalVector(urgency=0.0),
    )
    system = AttentionSystem()
    accepted = system.refresh((original,), event)
    before = accepted.snapshot.canonical_bytes()
    changed_signal = _goal_projection(
        event,
        signals=AttentionSignalVector(urgency=0.25),
    )
    changed_render = _goal_projection(event, rendered_bytes=65)
    changed_missingness = _goal_projection(
        event,
        signals=AttentionSignalVector(urgency=None),
    )
    changed_emotion = GlobalEmotionProjection._create(event=event, arousal=0.5)

    for projections, emotion in (
        ((changed_signal,), None),
        ((changed_render,), None),
        ((changed_missingness,), None),
        ((original,), changed_emotion),
    ):
        with pytest.raises((TypeError, ValueError)):
            system.refresh(projections, event, global_emotion=emotion)
        assert system.snapshot().canonical_bytes() == before

    conflicting_time = AttentionEvent(
        event.event_id,
        event.event_sequence,
        event.occurred_at + timedelta(seconds=1),
    )
    conflicting_projection = _goal_projection(conflicting_time)
    with pytest.raises((TypeError, ValueError)):
        system.refresh((conflicting_projection,), conflicting_time)
    assert system.snapshot().canonical_bytes() == before

    reused_id = AttentionEvent(
        event.event_id,
        event.event_sequence + 1,
        event.occurred_at + timedelta(seconds=2),
    )
    with pytest.raises((TypeError, ValueError)):
        system.refresh((_goal_projection(reused_id),), reused_id)
    assert system.snapshot().canonical_bytes() == before


def test_historical_exact_retry_returns_current_state_and_original_accepted_receipt() -> None:
    system = AttentionSystem()
    first_event = _event(1)
    historical_input = _goal_projection(
        first_event, source_revision=1, source_digest="a" * 64
    )
    first = system.refresh((historical_input,), first_event)
    later_event = _event(2)
    later_input = _goal_projection(
        later_event, source_revision=6, source_digest="b" * 64
    )
    later = system.refresh((later_input,), later_event)
    current_bytes = later.snapshot.canonical_bytes()

    historical_retry = system.refresh((historical_input,), first_event)

    assert historical_retry.replayed is True
    assert historical_retry.snapshot.canonical_bytes() == current_bytes
    assert historical_retry.snapshot.last_event == later_event
    assert historical_retry.receipt == first.receipt
    assert historical_retry.receipt.result_state_digest == first.snapshot.state_digest
    assert historical_retry.receipt.result_state_digest != later.snapshot.state_digest
    assert historical_retry.view == later.view
    assert system.snapshot().canonical_bytes() == current_bytes


def test_constructor_and_restore_retry_without_reranking_or_habituation_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suzka.attention.system as system_module

    event = _event(1)
    projection = _goal_projection(event)
    accepted = AttentionSystem().refresh((projection,), event)
    restored_from_constructor = AttentionSystem(accepted.snapshot)
    restored_from_restore = AttentionSystem()
    restored_from_restore.restore_snapshot(accepted.snapshot)

    for restored in (restored_from_constructor, restored_from_restore):
        selected = restored.selected_view()
        assert selected.event == event
        assert selected.focused_targets == (projection.target,)
        assert selected.unfinished_targets == ()
        assert selected.competition is None
        assert selected.prompt is None

    def must_not_run(*args: object, **kwargs: object) -> object:
        pytest.fail("an exact restored retry reran policy or continuity arithmetic")

    monkeypatch.setattr(system_module, "compete_attention", must_not_run)
    monkeypatch.setattr(system_module, "next_habituation_units", must_not_run)

    constructor_retry = restored_from_constructor.refresh((projection,), event)
    restore_retry = restored_from_restore.refresh((projection,), event)

    for replay in (constructor_retry, restore_retry):
        assert replay.replayed is True
        assert replay.snapshot.canonical_bytes() == accepted.snapshot.canonical_bytes()
        assert replay.receipt == accepted.receipt
        assert replay.view.focused_targets == (projection.target,)
        assert replay.view.competition is None
        assert replay.view.prompt is None


def test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out() -> None:
    event = _event(1)
    projection = _goal_projection(event)
    result = AttentionSystem().refresh((projection,), event)
    system = AttentionSystem(result.snapshot)
    baseline = system.snapshot().canonical_bytes()
    first_snapshot = system.snapshot()
    second_snapshot = system.snapshot()
    assert first_snapshot is not second_snapshot
    assert first_snapshot.candidates[0] is not second_snapshot.candidates[0]
    object.__setattr__(first_snapshot.candidates[0], "habituation", 999_999)
    assert system.snapshot().canonical_bytes() == baseline

    first_view = system.selected_view()
    second_view = system.selected_view()
    assert first_view is not second_view
    object.__setattr__(first_view, "focused_targets", ())
    assert system.selected_view().focused_targets == (projection.target,)
    assert second_view.focused_targets == (projection.target,)

    canonical = system.snapshot().canonical_value()
    assert set(canonical) == {
        "authority_digest",
        "candidates",
        "focused_ids",
        "last_event",
        "policy_version",
        "receipt_anchor",
        "receipts",
        "revision",
        "revision_anchor",
        "revision_history",
        "schema_version",
        "state_digest",
        "unfinished_ids",
    }
    encoded = system.snapshot().canonical_bytes().decode("ascii")
    for forbidden in (
        "rendered_text",
        "rendered_content",
        "source_text",
        "policy_weights",
        "private subject",
        "goal description",
    ):
        assert forbidden not in encoded
    assert type(system.snapshot().receipts[0]) is AttentionEventReceipt


def test_prompt_and_focus_view_are_read_only_projections_not_persisted_source_rows() -> None:
    from test_attention_adapters import _item, _view

    event = _event(1)
    item = _item("attention-prompt-source")
    source_view = _view(
        item,
        content="private resolved content must not cross the system boundary",
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    projection = project_working_memory(item, revision=1, event=event, view=source_view)
    system = AttentionSystem()
    result = system.refresh((projection,), event)

    assert result.view.prompt is not None
    assert result.view.prompt.included_candidate_ids == (projection.candidate_id,)
    assert result.view.focused_targets == (projection.target,)
    assert "private resolved content must not cross the system boundary" not in repr(
        result.snapshot
    )
    assert "private resolved content must not cross the system boundary" not in repr(
        result.view
    )
    assert system.selected_view() == result.view
