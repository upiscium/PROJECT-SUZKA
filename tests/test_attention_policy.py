"""Fixed, stateless R14 Attention competition and prompt-budget tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import pytest

from suzka.attention.bounds import attention_candidate_capacity
from suzka.attention.common import (
    ATTENTION_FIXED_POINT_SCALE,
    ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
    ATTENTION_MAX_COUNTER,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_RENDERED_ITEM_BYTES,
    ATTENTION_PROMPT_BUDGET_BYTES,
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
    AttentionRevisionEvidence,
    AttentionSignalVector,
    AttentionSourceWitness,
    AttentionTarget,
    AttentionRevisionReason,
    attention_state_digest,
)
from suzka.attention.policy import (
    AttentionCandidateDecision,
    ATTENTION_POLICY_FOCUSED_BASE_BONUS_UNITS,
    ATTENTION_POLICY_HABITUATION_MAX_PENALTY_UNITS,
    ATTENTION_POLICY_INHIBITION_MAX_PENALTY_UNITS,
    ATTENTION_POLICY_MAX_SATIATION_PENALTY_UNITS,
    ATTENTION_POLICY_MAX_UNCERTAINTY_PENALTY_UNITS,
    ATTENTION_POLICY_NEUTRAL_UNITS,
    ATTENTION_POLICY_SCORE_MINIMUM_UNITS,
    ATTENTION_POLICY_STREAK_BONUS_MAX_EVENTS,
    ATTENTION_POLICY_STREAK_BONUS_PER_EVENT_UNITS,
    ATTENTION_POLICY_SWITCH_COST_UNITS,
    ATTENTION_POLICY_WEIGHTS,
    ATTENTION_PROMPT_AUTHORITY_INTRO,
    ATTENTION_PROMPT_FRAME_BYTES,
    ATTENTION_PROMPT_FRAME_TEXT,
    ATTENTION_PROMPT_SECTION_HEADERS,
    AttentionCandidateReason,
    AttentionCompetitionResult,
    AttentionCompetitionOutcome,
    AttentionContinuityBaseline,
    AttentionFallbackReason,
    AttentionPromptReason,
    AttentionSignalDimension,
    AttentionUnfinishedOverflowError,
    compete_attention,
    next_habituation_units,
    next_inhibition_units,
    next_streak_counts,
    select_attention_prompt,
)


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _event(
    event_id: str = "attention-current",
    sequence: int = 2,
    *,
    seconds: int = 1,
) -> AttentionEvent:
    return AttentionEvent(event_id, sequence, NOW + timedelta(seconds=seconds))


def _target(kind: AttentionTargetKind, reference: str) -> AttentionTarget:
    return AttentionTarget(kind, reference)


def _source(
    target: AttentionTarget,
    *,
    event: AttentionEvent | None,
    revision: int = 7,
) -> AttentionSourceWitness:
    source_kind = {
        AttentionTargetKind.WORKING_MEMORY: AttentionSourceKind.WORKING_MEMORY,
        AttentionTargetKind.MOTIVATION: AttentionSourceKind.MOTIVATION,
        AttentionTargetKind.GOAL: AttentionSourceKind.GOAL,
        AttentionTargetKind.COMMITMENT: AttentionSourceKind.COMMITMENT,
    }[target.kind]
    return AttentionSourceWitness(
        kind=source_kind,
        reference=target.reference,
        revision=revision,
        digest="a" * 64,
        digest_kind=(
            SourceDigestKind.ATTENTION_PROJECTION
            if target.kind is AttentionTargetKind.WORKING_MEMORY
            else SourceDigestKind.UPSTREAM_AUTHORITY
        ),
        target_kind=target.kind,
        target_reference=target.reference,
        source_event_id=None if event is None or target.kind is AttentionTargetKind.WORKING_MEMORY else event.event_id,
        source_event_sequence=None
        if event is None or target.kind is AttentionTargetKind.WORKING_MEMORY
        else event.event_sequence,
        source_occurred_at=None
        if event is None or target.kind is AttentionTargetKind.WORKING_MEMORY
        else event.occurred_at,
    )


def _projection(
    kind: AttentionTargetKind,
    reference: str,
    *,
    event: AttentionEvent | None = None,
    signals: AttentionSignalVector | None = None,
    availability: CandidateAvailability = CandidateAvailability.ELIGIBLE,
    rendered_bytes: int = 64,
) -> AttentionCandidateProjection:
    current_event = event or _event()
    target = _target(kind, reference)
    eligible = availability is CandidateAvailability.ELIGIBLE
    return AttentionCandidateProjection._create(
        target=target,
        source=_source(target, event=current_event),
        signals=signals or AttentionSignalVector(),
        event=current_event,
        availability=availability,
        rendered_bytes=rendered_bytes if eligible else None,
        rendered_digest=target.candidate_id if eligible else None,
    )


def _wm(
    reference_byte: str = "a",
    *,
    event: AttentionEvent | None = None,
    signals: AttentionSignalVector | None = None,
    availability: CandidateAvailability = CandidateAvailability.ELIGIBLE,
    rendered_bytes: int = 64,
) -> AttentionCandidateProjection:
    return _projection(
        AttentionTargetKind.WORKING_MEMORY,
        "wm-" + reference_byte * 64,
        event=event,
        signals=signals,
        availability=availability,
        rendered_bytes=rendered_bytes,
    )


def _goal(
    reference_byte: str = "b",
    *,
    event: AttentionEvent | None = None,
    signals: AttentionSignalVector | None = None,
    availability: CandidateAvailability = CandidateAvailability.ELIGIBLE,
    rendered_bytes: int = 64,
) -> AttentionCandidateProjection:
    return _projection(
        AttentionTargetKind.GOAL,
        reference_byte * 64,
        event=event,
        signals=signals,
        availability=availability,
        rendered_bytes=rendered_bytes,
    )


def _motivation(
    reference_byte: str = "c",
    *,
    event: AttentionEvent | None = None,
    signals: AttentionSignalVector | None = None,
    availability: CandidateAvailability = CandidateAvailability.ELIGIBLE,
    rendered_bytes: int = 64,
) -> AttentionCandidateProjection:
    return _projection(
        AttentionTargetKind.MOTIVATION,
        reference_byte * 64,
        event=event,
        signals=signals,
        availability=availability,
        rendered_bytes=rendered_bytes,
    )


def _prior(
    projections: tuple[AttentionCandidateProjection, ...],
    *,
    event: AttentionEvent | None = None,
    focused_ids: tuple[str, ...] = (),
    unfinished_ids: tuple[str, ...] = (),
    streaks: dict[str, tuple[int, int, int, int]] | None = None,
) -> AttentionContinuity:
    """Build a valid prior using the same strict one-revision proof chain."""

    previous_event = event or AttentionEvent("prior-event", 5, NOW)
    streaks = {} if streaks is None else streaks
    projection_by_id = {item.candidate_id: item for item in projections}
    all_ids = set(focused_ids) | set(unfinished_ids) | set(streaks)
    if not all_ids.issubset(projection_by_id):
        raise ValueError("prior helper only supports current projection references")
    candidates: list[AttentionCandidateContinuity] = []
    for candidate_id in sorted(all_ids):
        target = projection_by_id[candidate_id].target
        settings = streaks.get(candidate_id, (0, 0, 1 if candidate_id in focused_ids else 0, 0))
        habituation, inhibition, focused_count, unattended_count = settings
        candidates.append(
            AttentionCandidateContinuity(
                target=target,
                source=_source(target, event=None, revision=2),
                availability=CandidateAvailability.ELIGIBLE,
                habituation=habituation,
                inhibition=inhibition,
                focused_event_count=focused_count,
                unattended_event_count=unattended_count,
            )
        )
    current_candidates = tuple(candidates)
    focused = tuple(sorted(focused_ids))
    unfinished = tuple(sorted(unfinished_ids))
    prior_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=1,
        last_event=previous_event,
        candidates=current_candidates,
        focused_ids=focused,
        unfinished_ids=unfinished,
    )
    history = AttentionRevisionEvidence(
        revision=1,
        event=previous_event,
        previous_state_digest=AttentionContinuity.bootstrap().state_digest,
        state_digest=prior_digest,
        previous_revision_digest=None,
        focused_ids=focused,
        unfinished_ids=unfinished,
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    receipt = AttentionEventReceipt(
        event=previous_event,
        input_digest="d" * 64,
        result_state_digest=prior_digest,
    )
    return AttentionContinuity(
        revision=1,
        last_event=previous_event,
        candidates=current_candidates,
        focused_ids=focused,
        unfinished_ids=unfinished,
        revision_history=(history,),
        receipts=(receipt,),
    )


def _decision(
    result: AttentionCompetitionResult,
    projection: AttentionCandidateProjection,
) -> AttentionCandidateDecision:
    return next(
        item
        for item in result.decisions
        if item.candidate_id == projection.candidate_id
    )


def test_fixed_weights_exact_penalties_and_unknown_fallback_are_reviewed() -> None:
    assert tuple((item.value, weight) for item, weight in ATTENTION_POLICY_WEIGHTS) == (
        ("activation", 2),
        ("salience", 2),
        ("strength", 2),
        ("persistence", 1),
        ("urgency", 2),
        ("context_compatibility", 1),
    )
    assert sum(weight for _, weight in ATTENTION_POLICY_WEIGHTS) == 10
    assert ATTENTION_POLICY_NEUTRAL_UNITS == 500_000
    assert ATTENTION_POLICY_SCORE_MINIMUM_UNITS == 150_000
    assert ATTENTION_POLICY_MAX_SATIATION_PENALTY_UNITS == 200_000
    assert ATTENTION_POLICY_MAX_UNCERTAINTY_PENALTY_UNITS == 100_000
    assert ATTENTION_POLICY_STREAK_BONUS_MAX_EVENTS == 4
    assert ATTENTION_POLICY_STREAK_BONUS_PER_EVENT_UNITS == 12_500
    assert ATTENTION_POLICY_FOCUSED_BASE_BONUS_UNITS == 100_000
    assert ATTENTION_POLICY_SWITCH_COST_UNITS == 50_000

    event = _event()
    measured = _motivation(
        signals=AttentionSignalVector(
            strength=0.75,
            persistence=0.25,
            satiation=0.5,
            uncertainty=0.25,
        ),
        event=event,
    )
    result = compete_attention((measured,), AttentionContinuity.bootstrap(), event)
    decision = _decision(result, measured)
    # The weighted floor is 525000; measured penalties are 100000 + 25000.
    assert decision.base_score_units == 525_000
    assert decision.satiation_penalty_units == 100_000
    assert decision.uncertainty_penalty_units == 25_000
    assert decision.score_units == 400_000
    assert decision.missing_dimensions == (
        AttentionSignalDimension.ACTIVATION,
        AttentionSignalDimension.SALIENCE,
        AttentionSignalDimension.URGENCY,
        AttentionSignalDimension.CONTEXT_COMPATIBILITY,
    )
    assert decision.fallback_reasons == (
        AttentionFallbackReason.UNKNOWN_WEIGHTED_SIGNAL_USES_NEUTRAL,
    )


def test_binary_float_quantization_uses_exact_fraction_floor() -> None:
    event = _event()
    value = 0.1
    units = Fraction.from_float(value).numerator * ATTENTION_FIXED_POINT_SCALE
    units //= Fraction.from_float(value).denominator
    projection = _wm(
        signals=AttentionSignalVector(activation=value),
        event=event,
    )
    decision = _decision(
        compete_attention((projection,), AttentionContinuity.bootstrap(), event),
        projection,
    )
    # Activation has weight two; the other eight weighted units use neutral.
    assert decision.base_score_units == (2 * units + 8 * 500_000) // 10


def test_unknown_and_observed_zero_have_distinct_scores_and_digests() -> None:
    event = _event()
    unknown = _wm("a", event=event)
    observed_zero = _wm("a", event=event, signals=AttentionSignalVector(activation=0.0))
    prior = AttentionContinuity.bootstrap()
    unknown_result = compete_attention((unknown,), prior, event)
    zero_result = compete_attention((observed_zero,), prior, event)

    assert _decision(unknown_result, unknown).score_units == 500_000
    assert _decision(zero_result, observed_zero).score_units == 400_000
    assert unknown_result.input_digest != zero_result.input_digest
    assert unknown_result.result_digest != zero_result.result_digest
    assert unknown_result.canonical_bytes() != zero_result.canonical_bytes()
    assert unknown_result.decisions[0].missing_dimensions != zero_result.decisions[0].missing_dimensions


def test_target_kind_does_not_hard_override_fixed_numeric_competition() -> None:
    event = _event()
    weak_goal = _goal(
        signals=AttentionSignalVector(urgency=0.0),
        event=event,
    )
    strong_motivation = _motivation(
        signals=AttentionSignalVector(strength=1.0, persistence=1.0),
        event=event,
    )
    result = compete_attention(
        (weak_goal, strong_motivation),
        AttentionContinuity.bootstrap(),
        event,
    )
    assert result.focus_order[0] == strong_motivation.candidate_id
    assert _decision(result, strong_motivation).score_units > _decision(result, weak_goal).score_units


def test_ties_and_input_permutations_have_canonical_total_order_and_digest() -> None:
    event = _event()
    projections = (_wm("a", event=event), _goal("b", event=event), _motivation("c", event=event))
    prior = AttentionContinuity.bootstrap()
    forward = compete_attention(projections, prior, event)
    reverse = compete_attention(tuple(reversed(projections)), prior, event)

    assert forward.focus_order == tuple(sorted(item.candidate_id for item in projections))
    assert forward.canonical_bytes() == reverse.canonical_bytes()
    assert forward.input_digest == reverse.input_digest
    assert forward.result_digest == reverse.result_digest


def test_continuity_bonus_switch_cost_hysteresis_and_displaced_unfinished() -> None:
    event = _event(sequence=6)
    prior_event = AttentionEvent("prior-event", 5, NOW)
    old_focus = _goal("a", event=event, signals=AttentionSignalVector(urgency=0.0))
    newcomers = tuple(
        _wm(f"{index:064x}"[-1], event=event, signals=AttentionSignalVector(
            activation=1.0,
            salience=1.0,
            context_compatibility=1.0,
        ))
        for index in range(1, ATTENTION_MAX_FOCUS + 1)
    )
    prior = _prior(
        (old_focus,),
        event=prior_event,
        focused_ids=(old_focus.candidate_id,),
        streaks={old_focus.candidate_id: (0, 0, 4, 0)},
    )
    result = compete_attention((old_focus, *newcomers), prior, event)

    old_decision = _decision(result, old_focus)
    newcomer_decision = _decision(result, newcomers[0])
    assert old_decision.focus_bonus_units == (
        ATTENTION_POLICY_FOCUSED_BASE_BONUS_UNITS
        + ATTENTION_POLICY_STREAK_BONUS_MAX_EVENTS
        * ATTENTION_POLICY_STREAK_BONUS_PER_EVENT_UNITS
    )
    assert old_decision.score_units == 550_000
    assert newcomer_decision.switch_cost_units == ATTENTION_POLICY_SWITCH_COST_UNITS
    assert newcomer_decision.score_units == 700_000
    assert len(result.focus_order) == ATTENTION_MAX_FOCUS
    assert old_focus.candidate_id not in result.focused_ids
    assert result.unfinished_ids == (old_focus.candidate_id,)


def test_habituation_inhibition_and_new_continuity_baseline_are_explicit() -> None:
    event = _event(sequence=6)
    candidate = _motivation(event=event)
    old = _prior(
        (candidate,),
        streaks={candidate.candidate_id: (500_000, 250_000, 0, 0)},
    )
    result = compete_attention((candidate,), old, event)
    decision = _decision(result, candidate)
    assert decision.habituation_penalty_units == (
        500_000 * ATTENTION_POLICY_HABITUATION_MAX_PENALTY_UNITS // ATTENTION_FIXED_POINT_SCALE
    )
    assert decision.inhibition_penalty_units == (
        250_000 * ATTENTION_POLICY_INHIBITION_MAX_PENALTY_UNITS // ATTENTION_FIXED_POINT_SCALE
    )
    assert decision.score_units == 300_000
    assert decision.continuity_baseline is AttentionContinuityBaseline.RETAINED
    assert decision.prior_candidate_present

    new_candidate = _motivation("d", event=event)
    new_result = compete_attention((new_candidate,), AttentionContinuity.bootstrap(), event)
    new_decision = _decision(new_result, new_candidate)
    assert new_decision.continuity_baseline is AttentionContinuityBaseline.NEW
    assert not new_decision.prior_candidate_present


def test_policy_is_idle_when_no_candidate_is_eligible_or_above_threshold() -> None:
    event = _event()
    empty = compete_attention((), AttentionContinuity.bootstrap(), event)
    assert empty.focused_ids == ()
    assert empty.outcome is AttentionCompetitionOutcome.IDLE_NO_ELIGIBLE_CANDIDATES

    unavailable = _wm(
        event=event,
        availability=CandidateAvailability.UNAVAILABLE,
    )
    no_eligible = compete_attention((unavailable,), AttentionContinuity.bootstrap(), event)
    assert no_eligible.focused_ids == ()
    assert no_eligible.outcome is AttentionCompetitionOutcome.IDLE_NO_ELIGIBLE_CANDIDATES
    assert _decision(no_eligible, unavailable).reason is AttentionCandidateReason.SOURCE_UNAVAILABLE

    below_threshold = _motivation(
        event=event,
        signals=AttentionSignalVector(
            strength=0.0,
            persistence=0.0,
            satiation=1.0,
            uncertainty=1.0,
        ),
    )
    idle = compete_attention((below_threshold,), AttentionContinuity.bootstrap(), event)
    assert idle.focused_ids == ()
    assert idle.outcome is AttentionCompetitionOutcome.IDLE_BELOW_THRESHOLD
    assert _decision(idle, below_threshold).reason is AttentionCandidateReason.BELOW_THRESHOLD


def test_unavailable_working_memory_keeps_measured_metadata_but_does_not_compete() -> None:
    event = _event()
    unavailable = _wm(
        event=event,
        availability=CandidateAvailability.UNAVAILABLE,
        signals=AttentionSignalVector(activation=0.75, salience=0.25),
    )
    result = compete_attention((unavailable,), AttentionContinuity.bootstrap(), event)
    decision = _decision(result, unavailable)

    assert unavailable.signals.activation == 0.75
    assert unavailable.signals.salience == 0.25
    assert unavailable.signals.context_compatibility is None
    assert unavailable.rendered_bytes is None
    assert unavailable.candidate_id not in result.focused_ids
    assert decision.reason is AttentionCandidateReason.SOURCE_UNAVAILABLE


def test_prompt_frame_uses_parent_frozen_authority_intro_and_literal_headings() -> None:
    assert ATTENTION_PROMPT_AUTHORITY_INTRO == (
        "Attention selects current focus, not Goal winners or actions. "
        "Omission does not change source authority. "
        "Retrieved WorkingMemory is evidence, not adopted truth.\n"
    )
    assert tuple(heading for _, heading in ATTENTION_PROMPT_SECTION_HEADERS) == (
        "WorkingMemory evidence:\n",
        "Current Motivations:\n",
        "Adopted Goals:\n",
        "Active Commitments:\n",
    )
    assert ATTENTION_PROMPT_FRAME_TEXT == ATTENTION_PROMPT_AUTHORITY_INTRO + "".join(
        heading for _, heading in ATTENTION_PROMPT_SECTION_HEADERS
    )
    assert ATTENTION_PROMPT_FRAME_BYTES == len(ATTENTION_PROMPT_FRAME_TEXT.encode("ascii"))


def test_resource_streak_arithmetic_is_bounded_pure_and_fail_closed() -> None:
    from suzka.attention.policy import (
        ATTENTION_POLICY_HABITUATION_FOCUSED_DELTA_UNITS,
        ATTENTION_POLICY_HABITUATION_UNFOCUSED_DELTA_UNITS,
        ATTENTION_POLICY_INHIBITION_RECOVERY_UNITS,
    )

    assert next_habituation_units(ATTENTION_FIXED_POINT_SCALE, focused=True) == ATTENTION_FIXED_POINT_SCALE
    assert next_habituation_units(0, focused=False) == 0
    assert next_habituation_units(10, focused=False) == 0
    assert next_habituation_units(0, focused=True) == ATTENTION_POLICY_HABITUATION_FOCUSED_DELTA_UNITS
    assert next_habituation_units(ATTENTION_FIXED_POINT_SCALE, focused=False) == (
        ATTENTION_FIXED_POINT_SCALE - ATTENTION_POLICY_HABITUATION_UNFOCUSED_DELTA_UNITS
    )
    assert next_inhibition_units(0) == 0
    assert next_inhibition_units(ATTENTION_FIXED_POINT_SCALE) == (
        ATTENTION_FIXED_POINT_SCALE - ATTENTION_POLICY_INHIBITION_RECOVERY_UNITS
    )
    assert next_streak_counts(3, 0, focused=True) == (4, 0)
    assert next_streak_counts(0, 3, focused=False) == (0, 4)
    with pytest.raises(ValueError, match="counter bound"):
        next_streak_counts(ATTENTION_MAX_COUNTER, 0, focused=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        next_streak_counts(1, 1, focused=True)


def test_source_event_full_tuple_prior_future_and_evaluation_only_repeat_fences() -> None:
    event = _event(sequence=6)
    bootstrap = AttentionContinuity.bootstrap()
    future_projection = _wm(event=_event("future-source-event", 7, seconds=3))
    with pytest.raises(ValueError, match="exactly match"):
        compete_attention((future_projection,), bootstrap, event)

    current_projection = _wm(event=event)
    future_prior = _prior(
        (current_projection,),
        event=AttentionEvent("later-prior", 7, NOW + timedelta(seconds=4)),
    )
    with pytest.raises(ValueError, match="older than retained"):
        compete_attention((current_projection,), future_prior, event)

    current_prior_event = AttentionEvent("same-event", 6, event.occurred_at)
    prior_candidate = _wm(event=current_prior_event)
    same_prior = _prior(
        (prior_candidate,),
        event=current_prior_event,
        focused_ids=(prior_candidate.candidate_id,),
    )
    repeat_projection = _wm(event=current_prior_event)
    repeat = compete_attention((repeat_projection,), same_prior, current_prior_event)
    repeated = compete_attention((repeat_projection,), same_prior, current_prior_event)
    assert repeat.same_event_evaluation
    assert repeat.canonical_bytes() == repeated.canonical_bytes()
    assert same_prior.last_event == current_prior_event

    conflicting = AttentionEvent("different-id", 6, event.occurred_at)
    conflicting_projection = _wm(event=conflicting)
    with pytest.raises(ValueError, match="exact same event"):
        compete_attention((conflicting_projection,), same_prior, conflicting)


def test_full_supported_candidate_capacity_has_explicit_results_and_rejects_one_over() -> None:
    event = _event()
    full: list[AttentionCandidateProjection] = []
    for index in range(4_096):
        full.append(
            _projection(
                AttentionTargetKind.WORKING_MEMORY,
                f"wm-{index:064x}",
                event=event,
                availability=CandidateAvailability.UNAVAILABLE,
            )
        )
    for kind in (
        AttentionTargetKind.MOTIVATION,
        AttentionTargetKind.GOAL,
        AttentionTargetKind.COMMITMENT,
    ):
        for index in range(32):
            full.append(
                _projection(
                    kind,
                    f"{index:064x}",
                    event=event,
                    availability=CandidateAvailability.UNAVAILABLE,
                )
            )
    complete = tuple(full)
    assert attention_candidate_capacity() == 4_192
    result = compete_attention(complete, AttentionContinuity.bootstrap(), event)
    assert len(result.decisions) == 4_192
    assert len({item.candidate_id for item in result.decisions}) == 4_192
    assert result.focused_ids == ()
    assert result.outcome is AttentionCompetitionOutcome.IDLE_NO_ELIGIBLE_CANDIDATES

    one_over = _projection(
        AttentionTargetKind.WORKING_MEMORY,
        "wm-" + "f" * 64,
        event=event,
        availability=CandidateAvailability.UNAVAILABLE,
    )
    with pytest.raises(ValueError, match="full candidate bound"):
        compete_attention((*complete, one_over), AttentionContinuity.bootstrap(), event)


def test_per_kind_source_authority_bound_is_derived_not_clipped() -> None:
    event = _event()
    too_many_working_memory = tuple(
        _projection(
            AttentionTargetKind.WORKING_MEMORY,
            f"wm-{index:064x}",
            event=event,
            availability=CandidateAvailability.UNAVAILABLE,
        )
        for index in range(4_097)
    )
    with pytest.raises(ValueError, match="source authority bound"):
        compete_attention(too_many_working_memory, AttentionContinuity.bootstrap(), event)


def test_unfinished_overflow_returns_bounded_fail_closed_evidence_without_slicing() -> None:
    event = _event(sequence=6)
    old_focuses = tuple(
        _wm(
            f"{index + 1:064x}"[-1],
            event=event,
            signals=AttentionSignalVector(
                activation=0.0,
                salience=0.0,
                context_compatibility=0.0,
            ),
        )
        for index in range(16)
    )
    previous_pending = tuple(_goal(f"{index + 1:064x}"[-1], event=event) for index in range(16))
    all_projections = (*old_focuses, *previous_pending)
    old_focus_ids = tuple(item.candidate_id for item in old_focuses)
    unfinished_ids = tuple(item.candidate_id for item in previous_pending)
    prior = _prior(
        all_projections,
        event=AttentionEvent("prior-event", 5, NOW),
        focused_ids=old_focus_ids,
        unfinished_ids=unfinished_ids,
    )
    # Old focused Working Memory rows lose to the urgency-maximized Goals.  The
    # result may not discard either old unfinished evidence or displaced focus.
    high_goals = tuple(
        _goal(
            f"{index + 1:064x}"[-1],
            event=event,
            signals=AttentionSignalVector(urgency=1.0),
        )
        for index in range(16)
    )
    with pytest.raises(AttentionUnfinishedOverflowError) as caught:
        compete_attention((*old_focuses, *high_goals), prior, event)
    assert len(caught.value.candidate_ids) == 32
    assert caught.value.candidate_ids == tuple(sorted(set(caught.value.candidate_ids)))
    assert len(caught.value.decisions) == 32


def test_prompt_exact_byte_packing_continues_after_omission_and_covers_every_candidate() -> None:
    event = _event()
    seed = (_wm("a", event=event), _wm("b", event=event), _wm("c", event=event))
    ordered = tuple(sorted(seed, key=lambda item: item.candidate_id))
    rows = (
        _projection(
            ordered[0].target.kind,
            ordered[0].target.reference,
            event=event,
            rendered_bytes=ATTENTION_PROMPT_BUDGET_BYTES - ATTENTION_PROMPT_FRAME_BYTES - 10,
        ),
        _projection(
            ordered[1].target.kind,
            ordered[1].target.reference,
            event=event,
            rendered_bytes=20,
        ),
        _projection(
            ordered[2].target.kind,
            ordered[2].target.reference,
            event=event,
            rendered_bytes=4,
        ),
    )
    competition = compete_attention(rows, AttentionContinuity.bootstrap(), event)
    prompt = select_attention_prompt(competition, rows)
    reasons = {item.candidate_id: item.reason for item in prompt.decisions}

    assert prompt.frame_bytes == ATTENTION_PROMPT_FRAME_BYTES
    assert prompt.total_bytes == ATTENTION_PROMPT_BUDGET_BYTES - 4
    assert prompt.total_bytes <= ATTENTION_PROMPT_BUDGET_BYTES
    assert prompt.included_candidate_ids == (rows[0].candidate_id, rows[2].candidate_id)
    assert reasons[rows[0].candidate_id] is AttentionPromptReason.INCLUDED
    assert reasons[rows[1].candidate_id] is AttentionPromptReason.OMIT_BYTE_BUDGET
    assert reasons[rows[2].candidate_id] is AttentionPromptReason.INCLUDED
    assert len(prompt.decisions) == len(rows)
    assert not any(hasattr(item, "rendered_text") for item in prompt.decisions)


@pytest.mark.parametrize(
    "rendered_bytes",
    (ATTENTION_PROMPT_BUDGET_BYTES + 1, ATTENTION_MAX_RENDERED_ITEM_BYTES),
)
def test_single_oversized_prompt_row_is_explicitly_omitted_not_truncated(
    rendered_bytes: int,
) -> None:
    event = _event()
    too_large = _wm(
        event=event,
        rendered_bytes=rendered_bytes,
    )
    result = compete_attention((too_large,), AttentionContinuity.bootstrap(), event)
    prompt = select_attention_prompt(result, (too_large,))
    decision = prompt.decisions[0]
    assert decision.reason is AttentionPromptReason.OVER_BUDGET_SINGLE
    assert decision.rendered_bytes == rendered_bytes
    assert decision.rendered_digest == too_large.rendered_digest
    assert prompt.included_candidate_ids == ()
    assert prompt.total_bytes == prompt.frame_bytes


def test_prompt_rejects_a_projection_set_that_differs_from_competition_evidence() -> None:
    event = _event()
    selected = _wm("a", event=event)
    other = _wm("b", event=event)
    result = compete_attention((selected,), AttentionContinuity.bootstrap(), event)
    with pytest.raises(ValueError, match="complete competition projection set"):
        select_attention_prompt(result, (other,))
    changed = _wm("a", event=event, rendered_bytes=65)
    with pytest.raises(ValueError, match="exact competition source witness"):
        select_attention_prompt(result, (changed,))


def test_unknown_global_emotion_is_explicit_and_emotion_only_changes_capacity() -> None:
    from suzka.attention.adapters import project_global_emotion
    from suzka.body import EmotionState

    event = _event()
    projections = tuple(_wm(f"{index + 1:064x}"[-1], event=event) for index in range(9))
    prior = AttentionContinuity.bootstrap()
    unknown = compete_attention(projections, prior, event, global_emotion=None)
    low = project_global_emotion(EmotionState(arousal=0.749999), event=event)
    high = project_global_emotion(EmotionState(arousal=0.75), event=event)
    low_result = compete_attention(projections, prior, event, global_emotion=low)
    high_result = compete_attention(projections, prior, event, global_emotion=high)

    assert unknown.focus_capacity == ATTENTION_MAX_FOCUS
    assert not unknown.global_emotion_known
    assert unknown.global_emotion_arousal_units is None
    assert low_result.focus_capacity == ATTENTION_MAX_FOCUS
    assert high_result.focus_capacity == ATTENTION_HIGH_AROUSAL_MAX_FOCUS
    assert tuple(item.score_units for item in low_result.decisions) == tuple(
        item.score_units for item in high_result.decisions
    )

    wrong_event = _event("different-emotion-event", 3)
    mismatched = project_global_emotion(EmotionState(arousal=0.5), event=wrong_event)
    with pytest.raises(ValueError, match="exactly match"):
        compete_attention(projections, prior, event, global_emotion=mismatched)


def test_tampered_global_emotion_digest_fails_without_repair() -> None:
    from suzka.attention.adapters import project_global_emotion
    from suzka.body import EmotionState

    event = _event()
    projection = _wm(event=event)
    global_emotion = project_global_emotion(EmotionState(arousal=0.5), event=event)
    foreign_digest = "0" * 64
    object.__setattr__(global_emotion, "projection_digest", foreign_digest)

    with pytest.raises(ValueError, match="Global Emotion projection digest"):
        compete_attention(
            (projection,),
            AttentionContinuity.bootstrap(),
            event,
            global_emotion=global_emotion,
        )
    assert global_emotion.projection_digest == foreign_digest


def test_policy_and_prompt_are_source_immutable_and_revalidate_event_bindings() -> None:
    event = _event()
    projection = _wm(event=event)
    before = projection.canonical_value()
    result = compete_attention((projection,), AttentionContinuity.bootstrap(), event)
    prompt = select_attention_prompt(result, (projection,))
    assert projection.canonical_value() == before
    assert prompt.competition_digest == result.result_digest
    assert projection.rendered_bytes is not None
    assert prompt.total_bytes == prompt.frame_bytes + projection.rendered_bytes + 1


def test_tampered_projection_digest_fails_competition_and_prompt_without_repair() -> None:
    event = _event()
    for_prompt = _wm(event=event)
    valid_result = compete_attention(
        (for_prompt,), AttentionContinuity.bootstrap(), event
    )
    foreign_digest = "0" * 64
    object.__setattr__(for_prompt, "projection_digest", foreign_digest)

    with pytest.raises(ValueError, match="projection digest"):
        compete_attention((for_prompt,), AttentionContinuity.bootstrap(), event)
    assert for_prompt.projection_digest == foreign_digest

    with pytest.raises(ValueError, match="projection digest"):
        select_attention_prompt(valid_result, (for_prompt,))
    assert for_prompt.projection_digest == foreign_digest


def test_tampered_prior_candidate_digest_fails_clone_without_repair() -> None:
    event = _event(sequence=6)
    candidate = _wm(event=event)
    prior = _prior(
        (candidate,),
        event=AttentionEvent("prior-event", 5, NOW),
        focused_ids=(candidate.candidate_id,),
    )
    witness = prior.candidates[0]
    foreign_digest = "0" * 64
    object.__setattr__(witness, "record_digest", foreign_digest)

    with pytest.raises(ValueError, match="candidate record digest"):
        compete_attention((candidate,), prior, event)
    assert witness.record_digest == foreign_digest


def test_tampered_prior_receipt_digest_fails_clone_without_repair() -> None:
    event = _event(sequence=6)
    candidate = _wm(event=event)
    prior = _prior(
        (candidate,),
        event=AttentionEvent("prior-event", 5, NOW),
        focused_ids=(candidate.candidate_id,),
    )
    witness = prior.receipts[0]
    foreign_digest = "0" * 64
    object.__setattr__(witness, "receipt_digest", foreign_digest)

    with pytest.raises(ValueError, match="receipt digest"):
        compete_attention((candidate,), prior, event)
    assert witness.receipt_digest == foreign_digest


def test_published_competition_and_prompt_digests_reject_tampering_on_revalidation() -> None:
    event = _event()
    projection = _wm(event=event)
    competition = compete_attention(
        (projection,), AttentionContinuity.bootstrap(), event
    )
    foreign_digest = "f" * 64

    object.__setattr__(competition, "result_digest", foreign_digest)
    with pytest.raises(ValueError, match="competition result digest"):
        select_attention_prompt(competition, (projection,))
    assert competition.result_digest == foreign_digest

    # Rebuild valid published values, then forge only the prompt result witness.
    competition = compete_attention(
        (projection,), AttentionContinuity.bootstrap(), event
    )
    prompt = select_attention_prompt(competition, (projection,))
    object.__setattr__(prompt, "result_digest", foreign_digest)
    with pytest.raises(ValueError, match="prompt result digest"):
        prompt.canonical_bytes()
    assert prompt.result_digest == foreign_digest


def test_published_competition_digest_binds_nested_decision_fields() -> None:
    event = _event()
    projection = _wm(event=event)
    competition = compete_attention(
        (projection,), AttentionContinuity.bootstrap(), event
    )
    decision = competition.decisions[0]
    original_score = decision.score_units
    object.__setattr__(decision, "score_units", original_score + 1)

    with pytest.raises(ValueError, match="competition result digest"):
        select_attention_prompt(competition, (projection,))
    assert decision.score_units == original_score + 1


def test_fresh_policy_import_does_not_load_adapters_runtime_or_model_dependencies() -> None:
    code = """
import importlib
import sys
importlib.import_module('suzka.attention.policy')
for prefix in (
    'suzka.runtime',
    'suzka.models',
    'suzka.providers',
    'torch',
    'transformers',
    'suzka.scheduler',
):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
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
