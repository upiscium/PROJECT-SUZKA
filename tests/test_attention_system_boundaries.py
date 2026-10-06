"""Process-local R14 Attention retention, atomicity, and authority boundaries."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import pytest

import suzka.attention.system as system_module
from suzka.attention.adapters import GlobalEmotionProjection
from suzka.attention.bounds import (
    ATTENTION_STATE_MAX_VALUE_BYTES,
    attention_candidate_capacity_by_kind,
    derive_attention_schema_size_budget,
    maximum_attention_continuity_fixture,
)
from suzka.attention.common import (
    ATTENTION_MAX_COUNTER,
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_MAX_EVENT_SEQUENCE,
    ATTENTION_MAX_REVISION,
    ATTENTION_MAX_REVISION_HISTORY,
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
    AttentionRevisionAnchor,
    AttentionRevisionEvidence,
    AttentionSignalVector,
    AttentionSourceWitness,
    AttentionTarget,
    attention_state_digest,
)
from suzka.attention.policy import (
    AttentionCompetitionResult,
    AttentionUnfinishedOverflowError,
)
from suzka.attention.system import AttentionReplayConflict, AttentionSystem


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
_SOURCE_KIND = {
    AttentionTargetKind.WORKING_MEMORY: AttentionSourceKind.WORKING_MEMORY,
    AttentionTargetKind.MOTIVATION: AttentionSourceKind.MOTIVATION,
    AttentionTargetKind.GOAL: AttentionSourceKind.GOAL,
    AttentionTargetKind.COMMITMENT: AttentionSourceKind.COMMITMENT,
}


def _event(sequence: int, *, seconds: int | None = None) -> AttentionEvent:
    offset = min(sequence, 10_000) if seconds is None else seconds
    return AttentionEvent(
        f"attention-boundary-{sequence}",
        sequence,
        NOW + timedelta(seconds=offset),
    )


def _target(kind: AttentionTargetKind, index: int) -> AttentionTarget:
    reference = (
        f"wm-{index:064x}"
        if kind is AttentionTargetKind.WORKING_MEMORY
        else f"{index:064x}"
    )
    return AttentionTarget(kind, reference)


def _projection(
    event: AttentionEvent,
    *,
    kind: AttentionTargetKind = AttentionTargetKind.WORKING_MEMORY,
    index: int = 0,
    availability: CandidateAvailability = CandidateAvailability.ELIGIBLE,
    signals: AttentionSignalVector | None = None,
) -> AttentionCandidateProjection:
    target = _target(kind, index)
    source_event = None if kind is AttentionTargetKind.WORKING_MEMORY else event
    source = AttentionSourceWitness(
        kind=_SOURCE_KIND[kind],
        reference=target.reference,
        revision=1,
        digest=f"{index + 1:064x}",
        digest_kind=(
            SourceDigestKind.ATTENTION_PROJECTION
            if kind is AttentionTargetKind.WORKING_MEMORY
            else SourceDigestKind.UPSTREAM_AUTHORITY
        ),
        target_kind=kind,
        target_reference=target.reference,
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
        rendered_bytes=64 if eligible else None,
        rendered_digest=target.candidate_id if eligible else None,
    )


def _single_revision_snapshot(
    event: AttentionEvent,
    projection: AttentionCandidateProjection,
    *,
    focused_event_count: int = 1,
    input_digest: str = "c" * 64,
) -> AttentionContinuity:
    focused_ids = (projection.candidate_id,) if focused_event_count else ()
    candidate = AttentionCandidateContinuity(
        target=projection.target,
        source=projection.source,
        availability=projection.availability,
        focused_event_count=focused_event_count,
    )
    state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=1,
        last_event=event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        unfinished_ids=(),
    )
    history = AttentionRevisionEvidence(
        revision=1,
        event=event,
        previous_state_digest=AttentionContinuity.bootstrap().state_digest,
        state_digest=state_digest,
        previous_revision_digest=None,
        focused_ids=focused_ids,
        unfinished_ids=(),
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    receipt = AttentionEventReceipt(
        event=event,
        input_digest=input_digest,
        result_state_digest=state_digest,
    )
    return AttentionContinuity(
        revision=1,
        last_event=event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        revision_history=(history,),
        receipts=(receipt,),
    )


def _short_unanchored_receipt_snapshot() -> AttentionContinuity:
    """A legal revision-2 root with one unanchored receipt, per the U1 contract."""

    first_event = _event(1)
    current_event = _event(2)
    current_projection = _projection(current_event)
    candidate = AttentionCandidateContinuity(
        target=current_projection.target,
        source=current_projection.source,
        availability=current_projection.availability,
        focused_event_count=1,
    )
    focused_ids = (candidate.candidate_id,)
    current_state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=2,
        last_event=current_event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        unfinished_ids=(),
    )
    first_history = AttentionRevisionEvidence(
        revision=1,
        event=first_event,
        previous_state_digest=AttentionContinuity.bootstrap().state_digest,
        state_digest="d" * 64,
        previous_revision_digest=None,
        focused_ids=focused_ids,
        unfinished_ids=(),
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    current_history = AttentionRevisionEvidence(
        revision=2,
        event=current_event,
        previous_state_digest=first_history.state_digest,
        state_digest=current_state_digest,
        previous_revision_digest=first_history.record_digest,
        focused_ids=focused_ids,
        unfinished_ids=(),
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    current_receipt = AttentionEventReceipt(
        event=current_event,
        input_digest="e" * 64,
        result_state_digest=current_state_digest,
    )
    return AttentionContinuity(
        revision=2,
        last_event=current_event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        revision_history=(first_history, current_history),
        receipts=(current_receipt,),
    )


def _maximum_revision_snapshot() -> AttentionContinuity:
    """Build a valid max-revision root whose event sequence still has headroom."""

    retained_revisions = ATTENTION_MAX_REVISION_HISTORY
    first_sequence = 1_001
    anchor_event = _event(first_sequence - 1, seconds=-1)
    current_event = _event(
        first_sequence + retained_revisions - 1,
        seconds=retained_revisions - 1,
    )
    current_state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=ATTENTION_MAX_REVISION,
        last_event=current_event,
        candidates=(),
        focused_ids=(),
        unfinished_ids=(),
    )
    anchor = AttentionRevisionAnchor(
        through_revision=ATTENTION_MAX_REVISION - retained_revisions,
        through_event=anchor_event,
        through_state_digest="a" * 64,
        through_revision_digest="b" * 64,
    )
    history: list[AttentionRevisionEvidence] = []
    receipts: list[AttentionEventReceipt] = []
    previous_revision_digest: str | None = anchor.through_revision_digest
    previous_state_digest = anchor.through_state_digest
    previous_receipt_digest: str | None = None
    for index in range(retained_revisions):
        revision = ATTENTION_MAX_REVISION - retained_revisions + 1 + index
        event = _event(first_sequence + index, seconds=index)
        state_digest = (
            current_state_digest if index == retained_revisions - 1 else f"{index + 1:064x}"
        )
        evidence = AttentionRevisionEvidence(
            revision=revision,
            event=event,
            previous_state_digest=previous_state_digest,
            state_digest=state_digest,
            previous_revision_digest=previous_revision_digest,
            focused_ids=(),
            unfinished_ids=(),
            reason=AttentionRevisionReason.STATE_UPDATE,
        )
        receipt = AttentionEventReceipt(
            event=event,
            input_digest="c" * 64,
            result_state_digest=state_digest,
            previous_receipt_digest=previous_receipt_digest,
        )
        history.append(evidence)
        receipts.append(receipt)
        previous_revision_digest = evidence.record_digest
        previous_state_digest = state_digest
        previous_receipt_digest = receipt.receipt_digest
    return AttentionContinuity(
        revision=ATTENTION_MAX_REVISION,
        last_event=current_event,
        revision_history=tuple(history),
        receipts=tuple(receipts),
        revision_anchor=anchor,
    )


def _maximum_streak_snapshot() -> AttentionContinuity:
    event = _event(ATTENTION_MAX_EVENT_SEQUENCE - 1, seconds=0)
    projection = _projection(event)
    return _single_revision_snapshot(
        event,
        projection,
        focused_event_count=ATTENTION_MAX_COUNTER,
    )


def _assert_view_matches(
    snapshot: AttentionContinuity,
    view: system_module.AttentionSelectedView,
) -> None:
    assert view.revision == snapshot.revision
    assert view.event == snapshot.last_event
    assert view.state_digest == snapshot.state_digest
    assert view.authority_digest == snapshot.authority_digest
    assert tuple(target.candidate_id for target in view.focused_targets) == snapshot.focused_ids
    assert tuple(target.candidate_id for target in view.unfinished_targets) == snapshot.unfinished_ids


def test_independent_retention_fences_retry_and_cross_history_evidence() -> None:
    attention = AttentionSystem()
    for sequence in range(1, 300):
        event = _event(sequence)
        attention.refresh((_projection(event),), event)

    before_last = attention.snapshot()
    assert before_last.revision == 299
    evicted_revision = before_last.revision_history[0]
    evicted_receipt = before_last.receipts[0]
    last_event = _event(300)
    attention.refresh((_projection(last_event),), last_event)
    current = attention.snapshot()

    assert current.revision == 300
    assert len(current.revision_history) == ATTENTION_MAX_REVISION_HISTORY == 16
    assert len(current.receipts) == ATTENTION_MAX_EVENT_RECEIPTS == 256
    assert current.revision_anchor is not None
    assert current.receipt_anchor is not None
    assert current.revision_anchor.through_revision == evicted_revision.revision
    assert current.revision_anchor.through_event == evicted_revision.event
    assert current.revision_anchor.through_state_digest == evicted_revision.state_digest
    assert current.revision_anchor.through_revision_digest == evicted_revision.record_digest
    assert current.receipt_anchor.through_event == evicted_receipt.event
    assert (
        current.receipt_anchor.through_result_state_digest
        == evicted_receipt.result_state_digest
    )
    assert current.receipt_anchor.through_receipt_digest == evicted_receipt.receipt_digest
    assert (
        current.revision_anchor.through_event.event_sequence
        != current.receipt_anchor.through_event.event_sequence
    )

    revision_states = {
        (item.event.event_id, item.event.event_sequence): (item.event, item.state_digest)
        for item in current.revision_history
    }
    retained_receipts = {
        (item.event.event_id, item.event.event_sequence): item
        for item in current.receipts
    }
    assert set(revision_states).issubset(retained_receipts)
    for key, (event, state_digest) in revision_states.items():
        receipt = retained_receipts[key]
        assert receipt.event == event
        assert receipt.result_state_digest == state_digest
    assert current.revision_history[-1].state_digest == current.state_digest
    assert current.receipts[-1].result_state_digest == current.state_digest

    # A receipt retained in the tail, not either anchor, is the input proof for
    # historical retries.  The returned view remains the current view.
    retained_receipt = current.receipts[0]
    retained_view = attention.selected_view()

    def replay_policy_is_forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("a retained retry reran Attention policy")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(system_module, "compete_attention", replay_policy_is_forbidden)
        patch.setattr(system_module, "select_attention_prompt", replay_policy_is_forbidden)
        replay = attention.refresh(
            (_projection(retained_receipt.event),),
            retained_receipt.event,
        )
    assert replay.replayed
    assert replay.receipt == retained_receipt
    assert replay.snapshot == current
    assert replay.view == retained_view
    assert attention.snapshot() == current

    assert current.revision_anchor is not None
    revision_anchor_receipt = next(
        item for item in current.receipts if item.event == current.revision_anchor.through_event
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(system_module, "compete_attention", replay_policy_is_forbidden)
        patch.setattr(system_module, "select_attention_prompt", replay_policy_is_forbidden)
        replay_at_revision_anchor = attention.refresh(
            (_projection(revision_anchor_receipt.event),),
            revision_anchor_receipt.event,
        )
    assert replay_at_revision_anchor.replayed
    assert replay_at_revision_anchor.receipt == revision_anchor_receipt
    assert replay_at_revision_anchor.snapshot == current
    assert replay_at_revision_anchor.view == retained_view

    # Anchors fence old IDs/sequences but do not retain the evicted input digest.
    # No assertion here extends event-ID uniqueness beyond retained evidence.
    assert current.receipt_anchor is not None
    unknown_old_events = (
        current.receipt_anchor.through_event,
        _event(current.receipt_anchor.through_event.event_sequence - 1),
    )
    for old_event in unknown_old_events:
        before_snapshot = attention.snapshot()
        before_view = attention.selected_view()
        with pytest.raises(AttentionReplayConflict):
            attention.refresh((_projection(old_event),), old_event)
        assert attention.snapshot() == before_snapshot
        assert attention.selected_view() == before_view


def test_canonical_restore_and_local_restore_do_not_replay_policy() -> None:
    original = AttentionSystem()
    event = _event(1)
    projection = _projection(event)
    fresh = original.refresh((projection,), event)
    canonical = fresh.snapshot.canonical_bytes()
    restored_snapshot = AttentionContinuity.from_json(canonical)
    assert restored_snapshot == fresh.snapshot

    def policy_or_source_lookup_is_forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("restore performed policy ranking or a source lookup")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(system_module, "compete_attention", policy_or_source_lookup_is_forbidden)
        patch.setattr(
            system_module,
            "select_attention_prompt",
            policy_or_source_lookup_is_forbidden,
        )
        restored = AttentionSystem(restored_snapshot)
        restored_view = restored.selected_view()
        original_view = original.selected_view()
        assert restored.snapshot() == fresh.snapshot
        assert restored_view.revision == original_view.revision
        assert restored_view.event == original_view.event
        assert restored_view.state_digest == original_view.state_digest
        assert restored_view.authority_digest == original_view.authority_digest
        assert restored_view.focused_targets == original_view.focused_targets
        assert restored_view.unfinished_targets == original_view.unfinished_targets
        assert restored_view.competition is None
        assert restored_view.prompt is None
        _assert_view_matches(restored.snapshot(), restored_view)

        replay = restored.refresh((projection,), event)
        assert replay.replayed
        assert replay.receipt == fresh.receipt
        assert replay.snapshot == fresh.snapshot

        short_chain = _short_unanchored_receipt_snapshot()
        assert len(short_chain.receipts) < short_chain.revision
        assert short_chain.receipt_anchor is None
        restored.restore_snapshot(short_chain)
        assert restored.snapshot() == short_chain
        local_view = restored.selected_view()
        assert local_view.competition is None
        assert local_view.prompt is None
        _assert_view_matches(short_chain, local_view)


def test_input_digest_binds_event_sorted_projections_and_emotion_not_prior_authority() -> None:
    seed = AttentionSystem()
    for sequence in (1, 2):
        event = _event(sequence)
        seed.refresh((_projection(event),), event)
    baseline = seed.snapshot()

    changed_first_receipt = replace(baseline.receipts[0], input_digest="f" * 64)
    changed_second_receipt = replace(
        baseline.receipts[1],
        previous_receipt_digest=changed_first_receipt.receipt_digest,
    )
    alternate_authority = replace(
        baseline,
        receipts=(changed_first_receipt, changed_second_receipt),
    )
    assert alternate_authority.state_digest == baseline.state_digest
    assert alternate_authority.authority_digest != baseline.authority_digest

    current_event = _event(3)
    projections = (_projection(current_event, index=0), _projection(current_event, index=1))
    first = AttentionSystem(baseline).refresh(projections, current_event)
    second = AttentionSystem(alternate_authority).refresh(
        tuple(reversed(projections)),
        current_event,
    )
    assert first.receipt.input_digest == second.receipt.input_digest

    later_event = _event(4)
    later_projections = (
        _projection(later_event, index=0),
        _projection(later_event, index=1),
    )
    later = AttentionSystem(first.snapshot).refresh(later_projections, later_event)
    assert later.receipt.input_digest != first.receipt.input_digest

    emotion = GlobalEmotionProjection._create(event=current_event, arousal=0.5)
    emotion_system = AttentionSystem(baseline)
    with_emotion = emotion_system.refresh(
        projections,
        current_event,
        global_emotion=emotion,
    )
    assert with_emotion.receipt.input_digest != first.receipt.input_digest
    before = emotion_system.snapshot()
    with pytest.raises(AttentionReplayConflict):
        emotion_system.refresh(projections, current_event)
    assert emotion_system.snapshot() == before


def test_returned_values_and_forged_source_packets_are_detached() -> None:
    attention = AttentionSystem()
    first_event = _event(1)
    attention.refresh((_projection(first_event),), first_event)
    owned_snapshot = attention.snapshot()
    owned_view = attention.selected_view()
    expected_snapshot = attention.snapshot()
    expected_view = attention.selected_view()
    assert owned_snapshot.candidates
    assert owned_view.focused_targets

    object.__setattr__(owned_snapshot.candidates[0], "habituation", 123)
    object.__setattr__(owned_view.focused_targets[0], "reference", "wm-" + "e" * 64)
    assert attention.snapshot() == expected_snapshot
    assert attention.selected_view() == expected_view

    next_event = _event(2)
    forged = _projection(next_event)
    assert forged.rendered_bytes is not None
    object.__setattr__(forged, "rendered_bytes", forged.rendered_bytes + 1)
    before_snapshot = attention.snapshot()
    before_view = attention.selected_view()
    with pytest.raises(ValueError, match="projection digest"):
        attention.refresh((forged,), next_event)
    assert attention.snapshot() == before_snapshot
    assert attention.selected_view() == before_view

    future_source_event = _event(3)
    forged_source = _projection(next_event, kind=AttentionTargetKind.GOAL)
    object.__setattr__(forged_source.source, "source_event_id", future_source_event.event_id)
    object.__setattr__(
        forged_source.source,
        "source_event_sequence",
        future_source_event.event_sequence,
    )
    object.__setattr__(
        forged_source.source,
        "source_occurred_at",
        future_source_event.occurred_at,
    )
    with pytest.raises(ValueError, match="future"):
        attention.refresh((forged_source,), next_event)
    assert attention.snapshot() == before_snapshot
    assert attention.selected_view() == before_view

    future_event = _event(3)
    future_projection = _projection(future_event, kind=AttentionTargetKind.GOAL)
    with pytest.raises(ValueError, match="projection events"):
        attention.refresh((future_projection,), next_event)
    assert attention.snapshot() == before_snapshot
    assert attention.selected_view() == before_view


@pytest.mark.parametrize(
    "failing_stage",
    ("compete_attention", "select_attention_prompt", "_copy_view"),
)
def test_every_injected_prepublication_failure_preserves_snapshot_and_view(
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
) -> None:
    attention = AttentionSystem()
    first_event = _event(1)
    attention.refresh((_projection(first_event),), first_event)
    before_snapshot = attention.snapshot()
    before_view = attention.selected_view()

    def fail_before_publication(*args: object, **kwargs: object) -> object:
        raise RuntimeError("static injected stage failure")

    monkeypatch.setattr(system_module, failing_stage, fail_before_publication)
    next_event = _event(2)
    with pytest.raises(RuntimeError, match="static injected stage failure"):
        attention.refresh((_projection(next_event),), next_event)

    # Undo the injection before using either public read; a failed stage did
    # not publish any component of its partially assembled result.
    monkeypatch.undo()
    assert attention.snapshot() == before_snapshot
    assert attention.selected_view() == before_view


def test_revision_and_streak_counter_overflow_fail_before_publication() -> None:
    revision_limited = AttentionSystem(_maximum_revision_snapshot())
    before_snapshot = revision_limited.snapshot()
    before_view = revision_limited.selected_view()
    next_event = _event(before_snapshot.last_event.event_sequence + 1)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="revision would exceed"):
        revision_limited.refresh((), next_event)
    assert revision_limited.snapshot() == before_snapshot
    assert revision_limited.selected_view() == before_view

    streak_limited = AttentionSystem(_maximum_streak_snapshot())
    before_snapshot = streak_limited.snapshot()
    before_view = streak_limited.selected_view()
    next_event = _event(ATTENTION_MAX_EVENT_SEQUENCE, seconds=1)
    with pytest.raises(ValueError, match="streak.*persisted counter bound"):
        streak_limited.refresh((_projection(next_event),), next_event)
    assert streak_limited.snapshot() == before_snapshot
    assert streak_limited.selected_view() == before_view


def test_full_candidate_universe_fits_and_one_over_bounds_fail_closed() -> None:
    event = _event(1)
    full: list[AttentionCandidateProjection] = []
    for kind_name, limit in attention_candidate_capacity_by_kind():
        kind = AttentionTargetKind(kind_name)
        full.extend(
            _projection(
                event,
                kind=kind,
                index=index,
                availability=CandidateAvailability.UNAVAILABLE,
            )
            for index in range(limit)
        )
    complete = tuple(full)
    attention = AttentionSystem()
    accepted = attention.refresh(complete, event)
    assert len(accepted.snapshot.candidates) == 4_192
    assert len({item.candidate_id for item in accepted.snapshot.candidates}) == 4_192

    before_snapshot = attention.snapshot()
    before_view = attention.selected_view()
    one_over = _projection(
        event,
        kind=AttentionTargetKind.WORKING_MEMORY,
        index=4_096,
        availability=CandidateAvailability.UNAVAILABLE,
    )
    with pytest.raises(ValueError, match="full candidate bound"):
        attention.refresh((*complete, one_over), event)
    assert attention.snapshot() == before_snapshot
    assert attention.selected_view() == before_view

    over_by_kind = (
        (AttentionTargetKind.WORKING_MEMORY, 4_097),
        (AttentionTargetKind.MOTIVATION, 33),
        (AttentionTargetKind.GOAL, 33),
        (AttentionTargetKind.COMMITMENT, 33),
    )
    for kind, count in over_by_kind:
        over_event = _event(1)
        projections = tuple(
            _projection(
                over_event,
                kind=kind,
                index=index,
                availability=CandidateAvailability.UNAVAILABLE,
            )
            for index in range(count)
        )
        fresh = AttentionSystem()
        untouched = fresh.snapshot()
        untouched_view = fresh.selected_view()
        with pytest.raises(ValueError, match="source authority bound"):
            fresh.refresh(projections, over_event)
        assert fresh.snapshot() == untouched
        assert fresh.selected_view() == untouched_view


def test_unfinished_reference_overflow_is_atomic_without_truncation() -> None:
    prior_event = _event(5)
    old_focus_projections = tuple(
        _projection(
            prior_event,
            kind=AttentionTargetKind.WORKING_MEMORY,
            index=index,
        )
        for index in range(16)
    )
    prior_pending_projections = tuple(
        _projection(
            prior_event,
            kind=AttentionTargetKind.GOAL,
            index=index,
        )
        for index in range(16)
    )
    all_prior_projections = (*old_focus_projections, *prior_pending_projections)
    old_focus_ids = tuple(sorted(item.candidate_id for item in old_focus_projections))
    prior_unfinished_ids = tuple(
        sorted(item.candidate_id for item in prior_pending_projections)
    )

    def prior_source(item: AttentionCandidateProjection) -> AttentionSourceWitness:
        if item.target.kind is not AttentionTargetKind.GOAL:
            return item.source
        return AttentionSourceWitness(
            kind=item.source.kind,
            reference=item.source.reference,
            revision=0,
            digest="0" * 64,
            digest_kind=item.source.digest_kind,
            target_kind=item.source.target_kind,
            target_reference=item.source.target_reference,
        )

    prior_candidates = tuple(
        sorted(
            (
                AttentionCandidateContinuity(
                    target=item.target,
                    source=prior_source(item),
                    availability=item.availability,
                    focused_event_count=1 if item.candidate_id in old_focus_ids else 0,
                )
                for item in all_prior_projections
            ),
            key=lambda item: item.candidate_id,
        )
    )
    prior_state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=1,
        last_event=prior_event,
        candidates=prior_candidates,
        focused_ids=old_focus_ids,
        unfinished_ids=prior_unfinished_ids,
    )
    prior_history = AttentionRevisionEvidence(
        revision=1,
        event=prior_event,
        previous_state_digest=AttentionContinuity.bootstrap().state_digest,
        state_digest=prior_state_digest,
        previous_revision_digest=None,
        focused_ids=old_focus_ids,
        unfinished_ids=prior_unfinished_ids,
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    prior_receipt = AttentionEventReceipt(
        event=prior_event,
        input_digest="d" * 64,
        result_state_digest=prior_state_digest,
    )
    prior = AttentionContinuity(
        revision=1,
        last_event=prior_event,
        candidates=prior_candidates,
        focused_ids=old_focus_ids,
        unfinished_ids=prior_unfinished_ids,
        revision_history=(prior_history,),
        receipts=(prior_receipt,),
    )

    current_event = _event(6)
    old_focus = tuple(
        _projection(
            current_event,
            kind=AttentionTargetKind.WORKING_MEMORY,
            index=index,
            signals=AttentionSignalVector(
                activation=0.0,
                salience=0.0,
                context_compatibility=0.0,
            ),
        )
        for index in range(16)
    )
    incoming_goals = tuple(
        _projection(
            current_event,
            kind=AttentionTargetKind.GOAL,
            index=index,
            signals=AttentionSignalVector(urgency=1.0),
        )
        for index in range(16)
    )
    attention = AttentionSystem(prior)
    before_snapshot = attention.snapshot()
    before_view = attention.selected_view()
    with pytest.raises(AttentionUnfinishedOverflowError) as caught:
        attention.refresh((*old_focus, *incoming_goals), current_event)
    assert len(caught.value.candidate_ids) == 32
    assert attention.snapshot() == before_snapshot
    assert attention.selected_view() == before_view


def test_refresh_read_and_restore_are_serialized_as_complete_publications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = AttentionSystem()
    initial = attention.snapshot()
    event = _event(1)
    projection = _projection(event)
    entered_policy = Event()
    release_policy = Event()
    reader_started = Event()
    reader_done = Event()
    restore_started = Event()
    restore_done = Event()
    refresh_results: list[system_module.AttentionRefreshResult] = []
    read_results: list[AttentionContinuity] = []
    errors: list[BaseException] = []
    real_competition = system_module.compete_attention

    def blocked_competition(
        projections: tuple[AttentionCandidateProjection, ...],
        prior: AttentionContinuity,
        current_event: AttentionEvent,
        global_emotion: GlobalEmotionProjection | None = None,
    ) -> AttentionCompetitionResult:
        entered_policy.set()
        if not release_policy.wait(5):
            raise RuntimeError("concurrent test did not release policy stage")
        return real_competition(projections, prior, current_event, global_emotion)

    monkeypatch.setattr(system_module, "compete_attention", blocked_competition)

    def refresh_worker() -> None:
        try:
            refresh_results.append(attention.refresh((projection,), event))
        except BaseException as error:  # Preserve worker errors for the main test thread.
            errors.append(error)

    def read_worker() -> None:
        reader_started.set()
        try:
            read_results.append(attention.snapshot())
        except BaseException as error:
            errors.append(error)
        finally:
            reader_done.set()

    def restore_worker() -> None:
        restore_started.set()
        try:
            attention.restore_snapshot(initial)
        except BaseException as error:
            errors.append(error)
        finally:
            restore_done.set()

    refresher = Thread(target=refresh_worker)
    reader: Thread | None = None
    restorer: Thread | None = None
    refresher.start()
    try:
        assert entered_policy.wait(5)
        reader = Thread(target=read_worker)
        restorer = Thread(target=restore_worker)
        reader.start()
        restorer.start()
        assert reader_started.wait(5)
        assert restore_started.wait(5)
        assert not reader_done.wait(0.05)
        assert not restore_done.wait(0.05)
    finally:
        release_policy.set()
    refresher.join(5)
    if reader is not None:
        reader.join(5)
    if restorer is not None:
        restorer.join(5)

    assert not refresher.is_alive()
    assert reader is not None and not reader.is_alive()
    assert restorer is not None and not restorer.is_alive()
    assert not errors
    assert len(refresh_results) == 1
    published_snapshot = refresh_results[0].snapshot
    assert published_snapshot.revision == 1
    assert len(read_results) == 1
    assert read_results[0] in (initial, published_snapshot)

    final_snapshot = attention.snapshot()
    final_view = attention.selected_view()
    _assert_view_matches(final_snapshot, final_view)
    assert final_snapshot in (initial, published_snapshot)


def test_full_schema_fixture_fits_value_bound_and_system_import_stays_local() -> None:
    maximum = maximum_attention_continuity_fixture()
    encoded = maximum.canonical_bytes()
    assert len(encoded) == derive_attention_schema_size_budget().attention_state_max_bytes
    assert len(encoded) <= ATTENTION_STATE_MAX_VALUE_BYTES
    assert len(maximum.candidates) == 4_192
    assert len(maximum.revision_history) == ATTENTION_MAX_REVISION_HISTORY
    assert len(maximum.receipts) == ATTENTION_MAX_EVENT_RECEIPTS

    code = """
import importlib
import sys

module = importlib.import_module("suzka.attention.system")
system = module.AttentionSystem
allowed = {"refresh", "restore_snapshot", "selected_view", "snapshot"}
public = {name for name in vars(system) if not name.startswith("_")}
assert public == allowed, public
for prefix in (
    "suzka.runtime",
    "suzka.memory",
    "suzka.cognition",
    "suzka.models",
    "torch",
    "transformers",
    "suzka.scheduler",
):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
assert not hasattr(module, "AgentState")
for name in (
    "refocus",
    "defer",
    "ignore",
    "schedule",
    "tick",
    "assessment",
    "export_attention_state",
    "restore_ports",
):
    assert not hasattr(system, name), name
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
