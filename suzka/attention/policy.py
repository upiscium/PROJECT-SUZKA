"""Stateless, fixed R14 Attention competition and prompt byte selection.

Policy consumes the complete bounded set of sealed source projections and an
immutable continuity snapshot.  Source adapters own coherent full-universe
coverage; policy independently enforces unique typed IDs and source bounds.  It
only proposes a focus and bounded unfinished references: it never advances or
mutates Attention continuity.
No source text, model ranking, caller weights, clocks, or runtime state enter
this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Final

from suzka.attention.adapters import GlobalEmotionProjection
from suzka.attention.bounds import (
    attention_candidate_capacity,
    attention_candidate_capacity_by_kind,
)
from suzka.attention.common import (
    ATTENTION_FIXED_POINT_SCALE,
    ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
    ATTENTION_HIGH_AROUSAL_THRESHOLD,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_COUNTER,
    ATTENTION_POLICY_VERSION,
    ATTENTION_PROMPT_BUDGET_BYTES,
    AttentionTargetKind,
    CandidateAvailability,
    bounded_counter,
    bounded_fraction,
    canonical_json,
    digest_payload,
    exact_enum,
    validate_digest,
)
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionCandidateProjection,
    AttentionContinuity,
    AttentionEvent,
)


class _ClosedEnum(str, Enum):
    """Base for fixed policy evidence vocabularies."""


class AttentionSignalDimension(_ClosedEnum):
    ACTIVATION = "activation"
    SALIENCE = "salience"
    STRENGTH = "strength"
    PERSISTENCE = "persistence"
    URGENCY = "urgency"
    CONTEXT_COMPATIBILITY = "context_compatibility"
    SATIATION = "satiation"
    UNCERTAINTY = "uncertainty"


class AttentionFallbackReason(_ClosedEnum):
    UNKNOWN_WEIGHTED_SIGNAL_USES_NEUTRAL = "unknown_weighted_signal_uses_neutral"
    UNKNOWN_PENALTY_SIGNAL_OMITS_PENALTY = "unknown_penalty_signal_omits_penalty"


class AttentionCandidateReason(_ClosedEnum):
    SELECTED_FOCUS = "selected_focus"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_INACTIVE = "source_inactive"
    BELOW_THRESHOLD = "below_threshold"
    FOCUS_CAPACITY = "focus_capacity"


class AttentionContinuityBaseline(_ClosedEnum):
    NEW = "new_continuity_baseline"
    RETAINED = "retained_continuity"


class AttentionCompetitionOutcome(_ClosedEnum):
    FOCUS_ASSIGNED = "focus_assigned"
    IDLE_NO_ELIGIBLE_CANDIDATES = "idle_no_eligible_candidates"
    IDLE_BELOW_THRESHOLD = "idle_below_threshold"


class AttentionPromptReason(_ClosedEnum):
    INCLUDED = "included"
    NOT_FOCUSED = "not_focused"
    OMIT_BYTE_BUDGET = "omit_byte_budget"
    OVER_BUDGET_SINGLE = "over_budget_single"


ATTENTION_POLICY_SCALE: Final[int] = ATTENTION_FIXED_POINT_SCALE
ATTENTION_POLICY_NEUTRAL_UNITS: Final[int] = 500_000
ATTENTION_POLICY_SCORE_MINIMUM_UNITS: Final[int] = 150_000
_HIGH_AROUSAL_THRESHOLD_FRACTION: Final[Fraction] = Fraction.from_float(
    ATTENTION_HIGH_AROUSAL_THRESHOLD
)
ATTENTION_POLICY_HIGH_AROUSAL_THRESHOLD_UNITS: Final[int] = (
    _HIGH_AROUSAL_THRESHOLD_FRACTION.numerator
    * ATTENTION_FIXED_POINT_SCALE
    // _HIGH_AROUSAL_THRESHOLD_FRACTION.denominator
)
ATTENTION_POLICY_WEIGHT_TOTAL: Final[int] = 10
ATTENTION_POLICY_WEIGHTS: Final[tuple[tuple[AttentionSignalDimension, int], ...]] = (
    (AttentionSignalDimension.ACTIVATION, 2),
    (AttentionSignalDimension.SALIENCE, 2),
    (AttentionSignalDimension.STRENGTH, 2),
    (AttentionSignalDimension.PERSISTENCE, 1),
    (AttentionSignalDimension.URGENCY, 2),
    (AttentionSignalDimension.CONTEXT_COMPATIBILITY, 1),
)
ATTENTION_POLICY_MAX_SATIATION_PENALTY_UNITS: Final[int] = 200_000
ATTENTION_POLICY_MAX_UNCERTAINTY_PENALTY_UNITS: Final[int] = 100_000
ATTENTION_POLICY_FOCUSED_BASE_BONUS_UNITS: Final[int] = 100_000
ATTENTION_POLICY_STREAK_BONUS_PER_EVENT_UNITS: Final[int] = 12_500
ATTENTION_POLICY_STREAK_BONUS_MAX_EVENTS: Final[int] = 4
ATTENTION_POLICY_SWITCH_COST_UNITS: Final[int] = 50_000
ATTENTION_POLICY_HABITUATION_MAX_PENALTY_UNITS: Final[int] = 150_000
ATTENTION_POLICY_INHIBITION_MAX_PENALTY_UNITS: Final[int] = 500_000
ATTENTION_POLICY_HABITUATION_FOCUSED_DELTA_UNITS: Final[int] = 50_000
ATTENTION_POLICY_HABITUATION_UNFOCUSED_DELTA_UNITS: Final[int] = 25_000
ATTENTION_POLICY_INHIBITION_RECOVERY_UNITS: Final[int] = 25_000
ATTENTION_POLICY_MAX_UNFINISHED: Final[int] = ATTENTION_MAX_FOCUS

# The frame is deliberately fixed ASCII.  These exact literals, not an
# estimated prompt reserve, form the framing portion of prompt byte accounting.
ATTENTION_PROMPT_AUTHORITY_INTRO: Final[str] = (
    "Attention selects current focus, not Goal winners or actions. "
    "Omission does not change source authority. "
    "Retrieved WorkingMemory is evidence, not adopted truth.\n"
)
ATTENTION_PROMPT_SECTION_HEADERS: Final[tuple[tuple[AttentionTargetKind, str], ...]] = (
    (AttentionTargetKind.WORKING_MEMORY, "WorkingMemory evidence:\n"),
    (AttentionTargetKind.MOTIVATION, "Current Motivations:\n"),
    (AttentionTargetKind.GOAL, "Adopted Goals:\n"),
    (AttentionTargetKind.COMMITMENT, "Active Commitments:\n"),
)
ATTENTION_PROMPT_FRAME_TEXT: Final[str] = ATTENTION_PROMPT_AUTHORITY_INTRO + "".join(
    heading for _, heading in ATTENTION_PROMPT_SECTION_HEADERS
)
ATTENTION_PROMPT_FRAME_BYTES: Final[int] = len(ATTENTION_PROMPT_FRAME_TEXT.encode("ascii"))

_ATTENTION_INPUT_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R14:ATTENTION-POLICY-INPUT:V1\0"
_ATTENTION_RESULT_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R14:ATTENTION-POLICY-RESULT:V1\0"
_ATTENTION_PROMPT_INPUT_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R14:ATTENTION-PROMPT-INPUT:V1\0"
_ATTENTION_PROMPT_RESULT_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R14:ATTENTION-PROMPT-RESULT:V1\0"

_WEIGHTED_DIMENSIONS: Final[tuple[AttentionSignalDimension, ...]] = tuple(
    dimension for dimension, _ in ATTENTION_POLICY_WEIGHTS
)
_ALL_SIGNAL_DIMENSIONS: Final[tuple[AttentionSignalDimension, ...]] = (
    *_WEIGHTED_DIMENSIONS,
    AttentionSignalDimension.SATIATION,
    AttentionSignalDimension.UNCERTAINTY,
)
_SIGNAL_DIMENSION_ORDER: Final[dict[str, int]] = {
    dimension.value: index
    for index, dimension in enumerate(_ALL_SIGNAL_DIMENSIONS)
}


class AttentionUnfinishedOverflowError(ValueError):
    """Fail-closed bounded proposal when unfinished evidence exceeds its cap.

    No identities are silently evicted.  The proposed union is bounded by the
    previous 16 unfinished and 16 focused references, and all candidate
    decisions are retained on the exception for the caller's explicit handling.
    """

    def __init__(
        self,
        *,
        candidate_ids: tuple[str, ...],
        focused_ids: tuple[str, ...],
        decisions: tuple[AttentionCandidateDecision, ...],
        input_digest: str,
    ) -> None:
        if type(candidate_ids) is not tuple or candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("overflow candidate IDs must be sorted and unique")
        if len(candidate_ids) <= ATTENTION_POLICY_MAX_UNFINISHED:
            raise ValueError("unfinished overflow must exceed its declared bound")
        if len(candidate_ids) > 2 * ATTENTION_MAX_FOCUS:
            raise ValueError("unfinished overflow evidence exceeds its maximum bounded union")
        self.candidate_ids = candidate_ids
        self.focused_ids = focused_ids
        self.decisions = decisions
        self.input_digest = validate_digest(input_digest, "input_digest")
        super().__init__(
            "proposed unfinished references exceed the bounded Attention limit "
            f"of {ATTENTION_POLICY_MAX_UNFINISHED}; no references were dropped"
        )


def _fixed_point_units(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= ATTENTION_POLICY_SCALE:
        raise ValueError(
            f"{name} must be an exact integer in [0, {ATTENTION_POLICY_SCALE}]"
        )
    return value


def _copy_attention_event(value: AttentionEvent) -> AttentionEvent:
    """Validate caller-owned event fields by constructing a separate value."""

    if type(value) is not AttentionEvent:
        raise TypeError("event must be an exact AttentionEvent")
    return AttentionEvent(value.event_id, value.event_sequence, value.occurred_at)


def _quantize_fraction(value: object, name: str) -> int:
    """Quantize one validated source float with exact binary-rational flooring."""

    fraction = bounded_fraction(value, name)
    exact = Fraction.from_float(fraction)
    return exact.numerator * ATTENTION_POLICY_SCALE // exact.denominator


def next_habituation_units(current: int, *, focused: bool) -> int:
    """Return bounded habituation arithmetic; this does not update continuity."""

    value = _fixed_point_units(current, "current habituation")
    if type(focused) is not bool:
        raise TypeError("focused must be an exact bool")
    delta = (
        ATTENTION_POLICY_HABITUATION_FOCUSED_DELTA_UNITS
        if focused
        else ATTENTION_POLICY_HABITUATION_UNFOCUSED_DELTA_UNITS
    )
    if focused:
        return min(ATTENTION_POLICY_SCALE, value + delta)
    return max(0, value - delta)


def next_inhibition_units(current: int) -> int:
    """Return one recovery-only inhibition arithmetic step."""

    value = _fixed_point_units(current, "current inhibition")
    return max(0, value - ATTENTION_POLICY_INHIBITION_RECOVERY_UNITS)


def next_streak_counts(
    focused_event_count: int,
    unattended_event_count: int,
    *,
    focused: bool,
) -> tuple[int, int]:
    """Return bounded current-streak arithmetic without clamping overflow."""

    focused_count = bounded_counter(focused_event_count, "focused_event_count")
    unattended_count = bounded_counter(unattended_event_count, "unattended_event_count")
    if type(focused) is not bool:
        raise TypeError("focused must be an exact bool")
    if focused_count and unattended_count:
        raise ValueError("focused and unattended streaks are mutually exclusive")
    if focused:
        if focused_count >= ATTENTION_MAX_COUNTER:
            raise ValueError("focused event streak would exceed its persisted counter bound")
        return focused_count + 1, 0
    if unattended_count >= ATTENTION_MAX_COUNTER:
        raise ValueError("unattended event streak would exceed its persisted counter bound")
    return 0, unattended_count + 1


def _global_emotion_evidence(
    value: GlobalEmotionProjection | None,
    event: AttentionEvent,
) -> tuple[str | None, int | None, bool, GlobalEmotionProjection | None]:
    if value is None:
        # Unknown emotion means the neutral resource ceiling, not measured zero.
        return None, None, False, None
    if type(value) is not GlobalEmotionProjection:
        raise TypeError("global_emotion must be an exact GlobalEmotionProjection or None")
    checked = value.validated_copy()
    projection_event = checked.event
    if type(projection_event) is not AttentionEvent or projection_event != event:
        raise ValueError("global emotion projection event must exactly match the Attention event")
    arousal = checked.arousal
    projection_digest = validate_digest(checked.projection_digest, "global emotion projection_digest")
    if arousal is None:
        return projection_digest, None, False, checked
    units = _quantize_fraction(arousal, "global emotion arousal")
    return projection_digest, units, True, checked


def _retained_events(prior: AttentionContinuity) -> tuple[AttentionEvent, ...]:
    events: list[AttentionEvent] = []
    if prior.last_event is not None:
        events.append(prior.last_event)
    events.extend(item.event for item in prior.revision_history)
    events.extend(item.event for item in prior.receipts)
    if prior.revision_anchor is not None:
        events.append(prior.revision_anchor.through_event)
    if prior.receipt_anchor is not None:
        events.append(prior.receipt_anchor.through_event)
    unique = {(
        item.event_id,
        item.event_sequence,
        item.occurred_at,
    ): item for item in events}
    return tuple(unique[key] for key in sorted(unique))


def _validate_attention_event(
    prior: AttentionContinuity,
    event: AttentionEvent,
) -> bool:
    """Validate a fresh event or the exact last event's evaluation-only repeat."""

    repeat_evaluation = False
    if prior.last_event is not None:
        if event.event_sequence < prior.last_event.event_sequence:
            raise ValueError("Attention event is older than retained continuity")
        if event.event_sequence == prior.last_event.event_sequence:
            if event != prior.last_event:
                raise ValueError("equal Attention event sequences must identify the exact same event")
            repeat_evaluation = True
        elif event.occurred_at < prior.last_event.occurred_at:
            raise ValueError("Attention event time is older than retained continuity")
    for retained in _retained_events(prior):
        if (
            retained.event_sequence == event.event_sequence
            or retained.event_id == event.event_id
        ) and retained != event:
            raise ValueError("Attention event ID or sequence conflicts with retained evidence")
    return repeat_evaluation


def _validate_projection_set(
    projections: tuple[AttentionCandidateProjection, ...],
    event: AttentionEvent,
) -> tuple[AttentionCandidateProjection, ...]:
    if type(projections) is not tuple:
        raise TypeError("projections must be an exact tuple of all current source projections")
    maximum = attention_candidate_capacity()
    if len(projections) > maximum:
        raise ValueError(f"source projection count exceeds the full candidate bound of {maximum}")
    per_kind_limits = dict(attention_candidate_capacity_by_kind())
    counts = {kind: 0 for kind in AttentionTargetKind}
    by_id: dict[str, AttentionCandidateProjection] = {}
    for source_projection in projections:
        if type(source_projection) is not AttentionCandidateProjection:
            raise TypeError("projections must contain exact sealed AttentionCandidateProjection values")
        projection = source_projection.validated_copy()
        if projection.event != event:
            raise ValueError("all source projection events must exactly match the supplied event")
        candidate_id = projection.candidate_id
        if candidate_id in by_id:
            raise ValueError("source projections must have unique canonical candidate IDs")
        by_id[candidate_id] = projection
        counts[projection.target.kind] += 1
    for kind, count in counts.items():
        if count > per_kind_limits[kind.value]:
            raise ValueError(
                f"{kind.value} source projection count exceeds its source authority bound"
            )
    return tuple(by_id[candidate_id] for candidate_id in sorted(by_id))


def _validate_prior(
    prior: AttentionContinuity,
) -> tuple[AttentionContinuity, dict[str, AttentionCandidateContinuity]]:
    if type(prior) is not AttentionContinuity:
        raise TypeError("prior must be an exact AttentionContinuity")
    # Canonical round-trip compares every published computed digest before
    # using the clone.  Calling __post_init__ on the input could repair stored
    # candidate/receipt hashes and erase evidence of post-publication mutation.
    validated_prior = AttentionContinuity.from_canonical_value(prior.canonical_value())
    prior_by_id: dict[str, AttentionCandidateContinuity] = {}
    for candidate in validated_prior.candidates:
        candidate.source.validate_primary_for_target(candidate.target)
        if validated_prior.last_event is not None:
            candidate.source.validate_for(candidate.target, validated_prior.last_event)
        if candidate.candidate_id != candidate.target.candidate_id:
            raise ValueError("prior candidate identity does not match its typed target")
        if candidate.candidate_id in prior_by_id:
            raise ValueError("prior candidate identities must be unique")
        prior_by_id[candidate.candidate_id] = candidate
    return validated_prior, prior_by_id


def _score_projection(
    projection: AttentionCandidateProjection,
    prior_candidate: AttentionCandidateContinuity | None,
    prior_focused_ids: frozenset[str],
    *,
    prior_has_focus: bool,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    tuple[AttentionSignalDimension, ...],
    tuple[AttentionFallbackReason, ...],
    AttentionContinuityBaseline,
]:
    signals = projection.signals
    missing_dimensions = tuple(
        dimension
        for dimension in _ALL_SIGNAL_DIMENSIONS
        if getattr(signals, dimension.value) is None
    )
    weighted_total = 0
    for dimension, weight in ATTENTION_POLICY_WEIGHTS:
        source_value = getattr(signals, dimension.value)
        units = (
            ATTENTION_POLICY_NEUTRAL_UNITS
            if source_value is None
            else _quantize_fraction(source_value, dimension.value)
        )
        weighted_total += weight * units
    base_score = weighted_total // ATTENTION_POLICY_WEIGHT_TOTAL

    satiation = signals.satiation
    uncertainty = signals.uncertainty
    satiation_penalty = (
        0
        if satiation is None
        else _quantize_fraction(satiation, "satiation")
        * ATTENTION_POLICY_MAX_SATIATION_PENALTY_UNITS
        // ATTENTION_POLICY_SCALE
    )
    uncertainty_penalty = (
        0
        if uncertainty is None
        else _quantize_fraction(uncertainty, "uncertainty")
        * ATTENTION_POLICY_MAX_UNCERTAINTY_PENALTY_UNITS
        // ATTENTION_POLICY_SCALE
    )

    prior_present = prior_candidate is not None
    baseline = (
        AttentionContinuityBaseline.RETAINED
        if prior_present
        else AttentionContinuityBaseline.NEW
    )
    habituation = 0 if prior_candidate is None else prior_candidate.habituation
    inhibition = 0 if prior_candidate is None else prior_candidate.inhibition
    habituation_penalty = (
        habituation * ATTENTION_POLICY_HABITUATION_MAX_PENALTY_UNITS
        // ATTENTION_POLICY_SCALE
    )
    inhibition_penalty = (
        inhibition * ATTENTION_POLICY_INHIBITION_MAX_PENALTY_UNITS
        // ATTENTION_POLICY_SCALE
    )

    candidate_id = projection.candidate_id
    focus_bonus = 0
    switch_cost = 0
    if candidate_id in prior_focused_ids:
        assert prior_candidate is not None
        focus_bonus = (
            ATTENTION_POLICY_FOCUSED_BASE_BONUS_UNITS
            + min(
                prior_candidate.focused_event_count,
                ATTENTION_POLICY_STREAK_BONUS_MAX_EVENTS,
            )
            * ATTENTION_POLICY_STREAK_BONUS_PER_EVENT_UNITS
        )
    else:
        unattended_streak = (
            0 if prior_candidate is None else prior_candidate.unattended_event_count
        )
        focus_bonus = (
            min(unattended_streak, ATTENTION_POLICY_STREAK_BONUS_MAX_EVENTS)
            * ATTENTION_POLICY_STREAK_BONUS_PER_EVENT_UNITS
        )
        if prior_has_focus:
            switch_cost = ATTENTION_POLICY_SWITCH_COST_UNITS

    unclamped = (
        base_score
        - satiation_penalty
        - uncertainty_penalty
        + focus_bonus
        - switch_cost
        - habituation_penalty
        - inhibition_penalty
    )
    score = min(ATTENTION_POLICY_SCALE, max(0, unclamped))
    fallback_reasons: list[AttentionFallbackReason] = []
    if any(
        getattr(signals, dimension.value) is None
        for dimension in _WEIGHTED_DIMENSIONS
    ):
        fallback_reasons.append(
            AttentionFallbackReason.UNKNOWN_WEIGHTED_SIGNAL_USES_NEUTRAL
        )
    if satiation is None or uncertainty is None:
        fallback_reasons.append(
            AttentionFallbackReason.UNKNOWN_PENALTY_SIGNAL_OMITS_PENALTY
        )
    return (
        base_score,
        satiation_penalty,
        uncertainty_penalty,
        focus_bonus,
        switch_cost,
        habituation_penalty,
        inhibition_penalty,
        score,
        missing_dimensions,
        tuple(fallback_reasons),
        baseline,
    )


@dataclass(frozen=True, slots=True)
class AttentionCandidateDecision:
    """One fixed-score outcome for every supplied current candidate."""

    candidate_id: str
    projection_digest: str
    availability: CandidateAvailability
    reason: AttentionCandidateReason
    base_score_units: int
    satiation_penalty_units: int
    uncertainty_penalty_units: int
    focus_bonus_units: int
    switch_cost_units: int
    habituation_penalty_units: int
    inhibition_penalty_units: int
    score_units: int
    missing_dimensions: tuple[AttentionSignalDimension, ...]
    fallback_reasons: tuple[AttentionFallbackReason, ...]
    continuity_baseline: AttentionContinuityBaseline
    prior_candidate_present: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", validate_digest(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self,
            "projection_digest",
            validate_digest(self.projection_digest, "projection_digest"),
        )
        exact_enum(self.availability, CandidateAvailability, "availability")
        exact_enum(self.reason, AttentionCandidateReason, "reason")
        exact_enum(self.continuity_baseline, AttentionContinuityBaseline, "continuity_baseline")
        if type(self.prior_candidate_present) is not bool:
            raise TypeError("prior_candidate_present must be an exact bool")
        if (self.continuity_baseline is AttentionContinuityBaseline.RETAINED) != self.prior_candidate_present:
            raise ValueError("continuity baseline must report whether a prior candidate existed")
        for name in (
            "base_score_units",
            "satiation_penalty_units",
            "uncertainty_penalty_units",
            "focus_bonus_units",
            "switch_cost_units",
            "habituation_penalty_units",
            "inhibition_penalty_units",
            "score_units",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        for name in ("base_score_units", "score_units"):
            _fixed_point_units(getattr(self, name), name)
        if type(self.missing_dimensions) is not tuple or any(
            type(item) is not AttentionSignalDimension for item in self.missing_dimensions
        ):
            raise TypeError("missing_dimensions must be an exact tuple of closed dimensions")
        if self.missing_dimensions != tuple(
            sorted(set(self.missing_dimensions), key=lambda item: _SIGNAL_DIMENSION_ORDER[item.value])
        ):
            raise ValueError("missing dimensions must use fixed unique policy order")
        if type(self.fallback_reasons) is not tuple or any(
            type(item) is not AttentionFallbackReason for item in self.fallback_reasons
        ):
            raise TypeError("fallback_reasons must be an exact tuple of closed reasons")
        if self.fallback_reasons != tuple(dict.fromkeys(self.fallback_reasons)):
            raise ValueError("fallback reasons must be unique and ordered")

    def canonical_value(self) -> dict[str, object]:
        return {
            "availability": self.availability.value,
            "base_score_units": self.base_score_units,
            "candidate_id": self.candidate_id,
            "continuity_baseline": self.continuity_baseline.value,
            "fallback_reasons": [item.value for item in self.fallback_reasons],
            "focus_bonus_units": self.focus_bonus_units,
            "habituation_penalty_units": self.habituation_penalty_units,
            "inhibition_penalty_units": self.inhibition_penalty_units,
            "missing_dimensions": [item.value for item in self.missing_dimensions],
            "prior_candidate_present": self.prior_candidate_present,
            "projection_digest": self.projection_digest,
            "reason": self.reason.value,
            "satiation_penalty_units": self.satiation_penalty_units,
            "score_units": self.score_units,
            "switch_cost_units": self.switch_cost_units,
            "uncertainty_penalty_units": self.uncertainty_penalty_units,
        }


@dataclass(frozen=True, slots=True)
class AttentionCompetitionResult:
    """Immutable full-universe competition proposal; it is not persisted state."""

    event: AttentionEvent
    input_digest: str
    prior_authority_digest: str
    global_emotion_projection_digest: str | None
    global_emotion_arousal_units: int | None
    global_emotion_known: bool
    focus_capacity: int
    outcome: AttentionCompetitionOutcome
    same_event_evaluation: bool
    focused_ids: tuple[str, ...]
    focus_order: tuple[str, ...]
    unfinished_ids: tuple[str, ...]
    decisions: tuple[AttentionCandidateDecision, ...]
    policy_version: int = ATTENTION_POLICY_VERSION
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        _copy_attention_event(self.event)
        object.__setattr__(self, "input_digest", validate_digest(self.input_digest, "input_digest"))
        object.__setattr__(
            self,
            "prior_authority_digest",
            validate_digest(self.prior_authority_digest, "prior_authority_digest"),
        )
        if self.global_emotion_projection_digest is not None:
            object.__setattr__(
                self,
                "global_emotion_projection_digest",
                validate_digest(
                    self.global_emotion_projection_digest,
                    "global_emotion_projection_digest",
                ),
            )
        if type(self.global_emotion_known) is not bool:
            raise TypeError("global_emotion_known must be an exact bool")
        if self.global_emotion_arousal_units is None:
            if self.global_emotion_known:
                raise ValueError("known global emotion requires quantized arousal units")
        else:
            if not self.global_emotion_known:
                raise ValueError("unknown global emotion cannot carry numeric arousal units")
            _fixed_point_units(self.global_emotion_arousal_units, "global emotion arousal")
        if type(self.focus_capacity) is not int or not 0 <= self.focus_capacity <= ATTENTION_MAX_FOCUS:
            raise ValueError("focus_capacity must be within the fixed Attention resource bound")
        exact_enum(self.outcome, AttentionCompetitionOutcome, "outcome")
        if type(self.same_event_evaluation) is not bool:
            raise TypeError("same_event_evaluation must be an exact bool")
        if type(self.policy_version) is not int or self.policy_version != ATTENTION_POLICY_VERSION:
            raise ValueError("unsupported Attention policy version")
        for name in ("focused_ids", "focus_order", "unfinished_ids", "decisions"):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
        decisions = self.decisions
        if any(type(item) is not AttentionCandidateDecision for item in decisions):
            raise TypeError("decisions must contain exact AttentionCandidateDecision values")
        for item in decisions:
            item.__post_init__()
        decision_ids = tuple(item.candidate_id for item in decisions)
        if decision_ids != tuple(sorted(set(decision_ids))):
            raise ValueError("candidate decisions must cover unique IDs in canonical order")
        for name in ("focused_ids", "unfinished_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
            for value in values:
                validate_digest(value, f"{name} item")
        if self.focus_order != tuple(dict.fromkeys(self.focus_order)):
            raise ValueError("focus_order must be unique")
        for candidate_id in self.focus_order:
            validate_digest(candidate_id, "focus_order item")
        if set(self.focused_ids) != set(self.focus_order):
            raise ValueError("focused IDs and ranked focus order must identify the same candidates")
        if len(self.focused_ids) > self.focus_capacity:
            raise ValueError("focused IDs exceed the current resource capacity")
        if len(self.unfinished_ids) > ATTENTION_POLICY_MAX_UNFINISHED:
            raise ValueError("unfinished IDs exceed their bounded policy limit")
        by_id = {item.candidate_id: item for item in decisions}
        if not set(self.focused_ids).issubset(by_id) or not set(self.unfinished_ids).issubset(by_id):
            raise ValueError("focus and unfinished references must resolve to current decisions")
        if any(
            by_id[candidate_id].reason is not AttentionCandidateReason.SELECTED_FOCUS
            for candidate_id in self.focus_order
        ):
            raise ValueError("focus order contains a candidate not selected by policy")
        ranked = tuple(
            item.candidate_id
            for item in sorted(decisions, key=lambda item: (-item.score_units, item.candidate_id))
            if item.reason is AttentionCandidateReason.SELECTED_FOCUS
        )
        if ranked != self.focus_order:
            raise ValueError("focus_order must use descending fixed score then candidate ID")
        if any(
            by_id[candidate_id].availability is not CandidateAvailability.ELIGIBLE
            for candidate_id in self.unfinished_ids
        ):
            raise ValueError("unfinished references must remain source-eligible")
        if not self.focused_ids and self.outcome is AttentionCompetitionOutcome.FOCUS_ASSIGNED:
            raise ValueError("focus-assigned outcome requires at least one focused candidate")
        if self.focused_ids and self.outcome is not AttentionCompetitionOutcome.FOCUS_ASSIGNED:
            raise ValueError("idle outcome cannot carry focused candidates")
        expected_digest = digest_payload(_ATTENTION_RESULT_DOMAIN, self._payload())
        if hasattr(self, "result_digest"):
            declared_digest = validate_digest(self.result_digest, "result_digest")
            if declared_digest != expected_digest:
                raise ValueError("Attention competition result digest does not match its contents")
        else:
            object.__setattr__(self, "result_digest", expected_digest)

    def _payload(self) -> dict[str, object]:
        return {
            "decisions": [item.canonical_value() for item in self.decisions],
            "event": self.event.canonical_value(),
            "focus_capacity": self.focus_capacity,
            "focus_order": list(self.focus_order),
            "focused_ids": list(self.focused_ids),
            "global_emotion_arousal_units": self.global_emotion_arousal_units,
            "global_emotion_known": self.global_emotion_known,
            "global_emotion_projection_digest": self.global_emotion_projection_digest,
            "input_digest": self.input_digest,
            "outcome": self.outcome.value,
            "policy_version": self.policy_version,
            "prior_authority_digest": self.prior_authority_digest,
            "same_event_evaluation": self.same_event_evaluation,
            "unfinished_ids": list(self.unfinished_ids),
        }

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._payload(), "result_digest": self.result_digest}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())


def compete_attention(
    projections: tuple[AttentionCandidateProjection, ...],
    prior: AttentionContinuity,
    event: AttentionEvent,
    global_emotion: GlobalEmotionProjection | None = None,
) -> AttentionCompetitionResult:
    """Compete the adapter-supplied full current source set with fixed weights.

    Input tuple order is not meaningful: candidate IDs are canonicalized before
    hashing, while focus order is score-descending with candidate-ID tie-break.
    This function validates bounds and uniqueness but cannot prove that an
    upstream adapter omitted no current target.
    Exact re-evaluation of ``prior.last_event`` is allowed for deterministic
    evaluation only; this pure function neither recognizes an idempotent state
    transition nor restores/replays prior focus.
    """

    if type(event) is not AttentionEvent:
        raise TypeError("event must be an exact AttentionEvent")
    event = _copy_attention_event(event)
    validated_prior, prior_by_id = _validate_prior(prior)
    same_event_evaluation = _validate_attention_event(validated_prior, event)
    ordered_projections = _validate_projection_set(projections, event)
    global_digest, global_arousal_units, global_known, checked_global_emotion = _global_emotion_evidence(
        global_emotion, event
    )
    capacity = (
        ATTENTION_MAX_FOCUS
        if not global_known
        else (
            ATTENTION_HIGH_AROUSAL_MAX_FOCUS
            if global_arousal_units is not None
            and global_arousal_units >= ATTENTION_POLICY_HIGH_AROUSAL_THRESHOLD_UNITS
            else ATTENTION_MAX_FOCUS
        )
    )
    prior_focused = frozenset(validated_prior.focused_ids)
    prior_authority_digest = validated_prior.authority_digest
    source_projection_values = [item.canonical_value() for item in ordered_projections]
    global_value = (
        None
        if checked_global_emotion is None
        else {
            "arousal": None
            if checked_global_emotion.arousal is None
            else checked_global_emotion.arousal.hex(),
            "event": checked_global_emotion.event.canonical_value(),
            "projection_digest": global_digest,
        }
    )
    input_digest = digest_payload(
        _ATTENTION_INPUT_DOMAIN,
        {
            "event": event.canonical_value(),
            "global_emotion": global_value,
            "policy_version": ATTENTION_POLICY_VERSION,
            "prior_authority_digest": prior_authority_digest,
            "projections": source_projection_values,
        },
    )

    calculated: list[tuple[AttentionCandidateProjection, AttentionCandidateDecision]] = []
    for projection in ordered_projections:
        previous = prior_by_id.get(projection.candidate_id)
        if previous is not None and previous.target != projection.target:
            raise ValueError("prior source metadata is bound to a different typed target")
        (
            base_score,
            satiation_penalty,
            uncertainty_penalty,
            focus_bonus,
            switch_cost,
            habituation_penalty,
            inhibition_penalty,
            score,
            missing_dimensions,
            fallback_reasons,
            baseline,
        ) = _score_projection(
            projection,
            previous,
            prior_focused,
            prior_has_focus=bool(validated_prior.focused_ids),
        )
        if projection.availability is CandidateAvailability.UNAVAILABLE:
            reason = AttentionCandidateReason.SOURCE_UNAVAILABLE
        elif projection.availability is CandidateAvailability.INACTIVE:
            reason = AttentionCandidateReason.SOURCE_INACTIVE
        elif score < ATTENTION_POLICY_SCORE_MINIMUM_UNITS:
            reason = AttentionCandidateReason.BELOW_THRESHOLD
        else:
            # Competition is assigned after all scores have been calculated.
            reason = AttentionCandidateReason.FOCUS_CAPACITY
        decision = AttentionCandidateDecision(
            candidate_id=projection.candidate_id,
            projection_digest=projection.projection_digest,
            availability=projection.availability,
            reason=reason,
            base_score_units=base_score,
            satiation_penalty_units=satiation_penalty,
            uncertainty_penalty_units=uncertainty_penalty,
            focus_bonus_units=focus_bonus,
            switch_cost_units=switch_cost,
            habituation_penalty_units=habituation_penalty,
            inhibition_penalty_units=inhibition_penalty,
            score_units=score,
            missing_dimensions=missing_dimensions,
            fallback_reasons=fallback_reasons,
            continuity_baseline=baseline,
            prior_candidate_present=previous is not None,
        )
        calculated.append((projection, decision))

    competitive = [
        item
        for item in calculated
        if item[0].availability is CandidateAvailability.ELIGIBLE
        and item[1].score_units >= ATTENTION_POLICY_SCORE_MINIMUM_UNITS
    ]
    competitive.sort(key=lambda pair: (-pair[1].score_units, pair[1].candidate_id))
    selected_pairs = competitive[:capacity]
    focus_order = tuple(decision.candidate_id for _, decision in selected_pairs)
    focused_ids = tuple(sorted(focus_order))
    selected_set = set(focused_ids)

    decisions: list[AttentionCandidateDecision] = []
    for projection, decision in calculated:
        if decision.reason is AttentionCandidateReason.FOCUS_CAPACITY and decision.candidate_id in selected_set:
            decision = AttentionCandidateDecision(
                candidate_id=decision.candidate_id,
                projection_digest=decision.projection_digest,
                availability=decision.availability,
                reason=AttentionCandidateReason.SELECTED_FOCUS,
                base_score_units=decision.base_score_units,
                satiation_penalty_units=decision.satiation_penalty_units,
                uncertainty_penalty_units=decision.uncertainty_penalty_units,
                focus_bonus_units=decision.focus_bonus_units,
                switch_cost_units=decision.switch_cost_units,
                habituation_penalty_units=decision.habituation_penalty_units,
                inhibition_penalty_units=decision.inhibition_penalty_units,
                score_units=decision.score_units,
                missing_dimensions=decision.missing_dimensions,
                fallback_reasons=decision.fallback_reasons,
                continuity_baseline=decision.continuity_baseline,
                prior_candidate_present=decision.prior_candidate_present,
            )
        decisions.append(decision)
    canonical_decisions = tuple(sorted(decisions, key=lambda item: item.candidate_id))

    projection_by_id = {item.candidate_id: item for item in ordered_projections}
    unfinished_set = {
        candidate_id
        for candidate_id in validated_prior.unfinished_ids
        if candidate_id in projection_by_id
        and projection_by_id[candidate_id].availability is CandidateAvailability.ELIGIBLE
    }
    unfinished_set.update(
        candidate_id
        for candidate_id in validated_prior.focused_ids
        if candidate_id not in selected_set
        and candidate_id in projection_by_id
        and projection_by_id[candidate_id].availability is CandidateAvailability.ELIGIBLE
    )
    proposed_unfinished = tuple(sorted(unfinished_set))
    if len(proposed_unfinished) > ATTENTION_POLICY_MAX_UNFINISHED:
        raise AttentionUnfinishedOverflowError(
            candidate_ids=proposed_unfinished,
            focused_ids=focused_ids,
            decisions=canonical_decisions,
            input_digest=input_digest,
        )

    if focused_ids:
        outcome = AttentionCompetitionOutcome.FOCUS_ASSIGNED
    elif not any(
        projection.availability is CandidateAvailability.ELIGIBLE
        for projection in ordered_projections
    ):
        outcome = AttentionCompetitionOutcome.IDLE_NO_ELIGIBLE_CANDIDATES
    else:
        outcome = AttentionCompetitionOutcome.IDLE_BELOW_THRESHOLD
    return AttentionCompetitionResult(
        event=event,
        input_digest=input_digest,
        prior_authority_digest=prior_authority_digest,
        global_emotion_projection_digest=global_digest,
        global_emotion_arousal_units=global_arousal_units,
        global_emotion_known=global_known,
        focus_capacity=capacity,
        outcome=outcome,
        same_event_evaluation=same_event_evaluation,
        focused_ids=focused_ids,
        focus_order=focus_order,
        unfinished_ids=proposed_unfinished,
        decisions=canonical_decisions,
    )


@dataclass(frozen=True, slots=True)
class AttentionPromptDecision:
    """Reference-only prompt inclusion or omission evidence for one candidate."""

    candidate_id: str
    projection_digest: str
    competition_reason: AttentionCandidateReason
    reason: AttentionPromptReason
    rendered_bytes: int | None
    rendered_digest: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", validate_digest(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self,
            "projection_digest",
            validate_digest(self.projection_digest, "projection_digest"),
        )
        exact_enum(self.competition_reason, AttentionCandidateReason, "competition_reason")
        exact_enum(self.reason, AttentionPromptReason, "reason")
        if self.rendered_bytes is None:
            if self.rendered_digest is not None:
                raise ValueError("a missing rendered row cannot carry a rendered digest")
        else:
            if type(self.rendered_bytes) is not int or self.rendered_bytes <= 0:
                raise ValueError("rendered_bytes must be a positive exact byte count")
            object.__setattr__(
                self,
                "rendered_digest",
                validate_digest(self.rendered_digest, "rendered_digest"),
            )
        if (self.reason is AttentionPromptReason.INCLUDED) != (self.rendered_bytes is not None):
            if self.reason is AttentionPromptReason.INCLUDED:
                raise ValueError("included prompt rows require exact source byte witnesses")
        if self.reason is AttentionPromptReason.INCLUDED and (
            self.competition_reason is not AttentionCandidateReason.SELECTED_FOCUS
        ):
            raise ValueError("only selected focus candidates may be included in the prompt")

    def canonical_value(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "competition_reason": self.competition_reason.value,
            "projection_digest": self.projection_digest,
            "reason": self.reason.value,
            "rendered_bytes": self.rendered_bytes,
            "rendered_digest": self.rendered_digest,
        }


@dataclass(frozen=True, slots=True)
class AttentionPromptSelection:
    """Exact fixed-frame and source-row byte witnesses, without raw source text."""

    competition_digest: str
    input_digest: str
    focus_capacity: int
    included_candidate_ids: tuple[str, ...]
    decisions: tuple[AttentionPromptDecision, ...]
    frame_bytes: int
    row_bytes: int
    newline_bytes: int
    total_bytes: int
    budget_bytes: int = ATTENTION_PROMPT_BUDGET_BYTES
    result_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "competition_digest",
            validate_digest(self.competition_digest, "competition_digest"),
        )
        object.__setattr__(self, "input_digest", validate_digest(self.input_digest, "input_digest"))
        if type(self.focus_capacity) is not int or not 0 <= self.focus_capacity <= ATTENTION_MAX_FOCUS:
            raise ValueError("focus_capacity must be within the Attention resource bound")
        if type(self.included_candidate_ids) is not tuple or type(self.decisions) is not tuple:
            raise TypeError("prompt IDs and decisions must be exact tuples")
        if self.included_candidate_ids != tuple(dict.fromkeys(self.included_candidate_ids)):
            raise ValueError("included candidate IDs must be unique and retain focus order")
        for candidate_id in self.included_candidate_ids:
            validate_digest(candidate_id, "included candidate ID")
        if any(type(item) is not AttentionPromptDecision for item in self.decisions):
            raise TypeError("decisions must contain exact AttentionPromptDecision values")
        for item in self.decisions:
            item.__post_init__()
        ids = tuple(item.candidate_id for item in self.decisions)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("prompt decisions must cover unique candidates in canonical ID order")
        if not set(self.included_candidate_ids).issubset(ids):
            raise ValueError("included prompt IDs must resolve to prompt decisions")
        included_from_decisions = {
            item.candidate_id
            for item in self.decisions
            if item.reason is AttentionPromptReason.INCLUDED
        }
        if included_from_decisions != set(self.included_candidate_ids):
            raise ValueError("included IDs must match explicit prompt inclusion decisions")
        if len(self.included_candidate_ids) > self.focus_capacity:
            raise ValueError("prompt inclusion exceeds the competition focus capacity")
        for name in ("frame_bytes", "row_bytes", "newline_bytes", "total_bytes", "budget_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact byte count")
        if self.frame_bytes != ATTENTION_PROMPT_FRAME_BYTES:
            raise ValueError("prompt frame byte count must come from the fixed ASCII frame")
        expected_rows = sum(
            item.rendered_bytes or 0
            for item in self.decisions
            if item.reason is AttentionPromptReason.INCLUDED
        )
        if self.row_bytes != expected_rows:
            raise ValueError("row byte total does not equal exact included source witnesses")
        if self.newline_bytes != len(self.included_candidate_ids):
            raise ValueError("prompt accounting requires exactly one newline per included row")
        if self.total_bytes != self.frame_bytes + self.row_bytes + self.newline_bytes:
            raise ValueError("total prompt bytes must equal frame, row, and newline witnesses")
        if self.budget_bytes != ATTENTION_PROMPT_BUDGET_BYTES or self.total_bytes > self.budget_bytes:
            raise ValueError("prompt selection exceeds the fixed total byte budget")
        expected_digest = digest_payload(_ATTENTION_PROMPT_RESULT_DOMAIN, self._payload())
        if hasattr(self, "result_digest"):
            declared_digest = validate_digest(self.result_digest, "result_digest")
            if declared_digest != expected_digest:
                raise ValueError("Attention prompt result digest does not match its contents")
        else:
            object.__setattr__(self, "result_digest", expected_digest)

    def _payload(self) -> dict[str, object]:
        return {
            "budget_bytes": self.budget_bytes,
            "competition_digest": self.competition_digest,
            "decisions": [item.canonical_value() for item in self.decisions],
            "focus_capacity": self.focus_capacity,
            "frame_bytes": self.frame_bytes,
            "included_candidate_ids": list(self.included_candidate_ids),
            "input_digest": self.input_digest,
            "newline_bytes": self.newline_bytes,
            "row_bytes": self.row_bytes,
            "total_bytes": self.total_bytes,
        }

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._payload(), "result_digest": self.result_digest}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())


def select_attention_prompt(
    competition: AttentionCompetitionResult,
    projections: tuple[AttentionCandidateProjection, ...],
) -> AttentionPromptSelection:
    """Pack focused source rows by fixed ranking and exact witnessed byte size."""

    if type(competition) is not AttentionCompetitionResult:
        raise TypeError("competition must be an exact AttentionCompetitionResult")
    competition.__post_init__()
    ordered_projections = _validate_projection_set(projections, competition.event)
    projection_by_id = {item.candidate_id: item for item in ordered_projections}
    decision_by_id = {item.candidate_id: item for item in competition.decisions}
    if set(projection_by_id) != set(decision_by_id):
        raise ValueError("prompt selection requires the complete competition projection set")
    for candidate_id, projection in projection_by_id.items():
        if projection.projection_digest != decision_by_id[candidate_id].projection_digest:
            raise ValueError("prompt projection differs from the exact competition source witness")

    total = ATTENTION_PROMPT_FRAME_BYTES
    row_total = 0
    newline_total = 0
    included: list[str] = []
    prompt_reasons: dict[str, AttentionPromptReason] = {}
    for candidate_id in competition.focus_order:
        projection = projection_by_id[candidate_id]
        if projection.availability is not CandidateAvailability.ELIGIBLE:
            raise ValueError("focused candidates must have source-eligible rendered rows")
        if type(projection.rendered_bytes) is not int or projection.rendered_digest is None:
            raise ValueError("focused candidates require exact rendered row witnesses")
        row_bytes = projection.rendered_bytes
        row_with_newline = row_bytes + 1
        if ATTENTION_PROMPT_FRAME_BYTES + row_with_newline > ATTENTION_PROMPT_BUDGET_BYTES:
            prompt_reasons[candidate_id] = AttentionPromptReason.OVER_BUDGET_SINGLE
        elif total + row_with_newline > ATTENTION_PROMPT_BUDGET_BYTES:
            prompt_reasons[candidate_id] = AttentionPromptReason.OMIT_BYTE_BUDGET
        else:
            prompt_reasons[candidate_id] = AttentionPromptReason.INCLUDED
            included.append(candidate_id)
            total += row_with_newline
            row_total += row_bytes
            newline_total += 1

    focused_set = set(competition.focused_ids)
    for candidate_id in projection_by_id:
        if candidate_id not in focused_set:
            prompt_reasons[candidate_id] = AttentionPromptReason.NOT_FOCUSED

    prompt_decisions = tuple(
        AttentionPromptDecision(
            candidate_id=candidate_id,
            projection_digest=projection_by_id[candidate_id].projection_digest,
            competition_reason=decision_by_id[candidate_id].reason,
            reason=prompt_reasons[candidate_id],
            rendered_bytes=projection_by_id[candidate_id].rendered_bytes,
            rendered_digest=projection_by_id[candidate_id].rendered_digest,
        )
        for candidate_id in sorted(projection_by_id)
    )
    prompt_input_digest = digest_payload(
        _ATTENTION_PROMPT_INPUT_DOMAIN,
        {
            "competition_digest": competition.result_digest,
            "projection_digests": [
                projection_by_id[candidate_id].projection_digest
                for candidate_id in sorted(projection_by_id)
            ],
        },
    )
    return AttentionPromptSelection(
        competition_digest=competition.result_digest,
        input_digest=prompt_input_digest,
        focus_capacity=competition.focus_capacity,
        included_candidate_ids=tuple(included),
        decisions=prompt_decisions,
        frame_bytes=ATTENTION_PROMPT_FRAME_BYTES,
        row_bytes=row_total,
        newline_bytes=newline_total,
        total_bytes=total,
    )


__all__ = [
    "ATTENTION_POLICY_FOCUSED_BASE_BONUS_UNITS",
    "ATTENTION_POLICY_HIGH_AROUSAL_THRESHOLD_UNITS",
    "ATTENTION_POLICY_HABITUATION_FOCUSED_DELTA_UNITS",
    "ATTENTION_POLICY_HABITUATION_MAX_PENALTY_UNITS",
    "ATTENTION_POLICY_HABITUATION_UNFOCUSED_DELTA_UNITS",
    "ATTENTION_POLICY_INHIBITION_MAX_PENALTY_UNITS",
    "ATTENTION_POLICY_INHIBITION_RECOVERY_UNITS",
    "ATTENTION_POLICY_MAX_SATIATION_PENALTY_UNITS",
    "ATTENTION_POLICY_MAX_UNFINISHED",
    "ATTENTION_POLICY_MAX_UNCERTAINTY_PENALTY_UNITS",
    "ATTENTION_POLICY_NEUTRAL_UNITS",
    "ATTENTION_POLICY_SCALE",
    "ATTENTION_POLICY_SCORE_MINIMUM_UNITS",
    "ATTENTION_POLICY_STREAK_BONUS_MAX_EVENTS",
    "ATTENTION_POLICY_STREAK_BONUS_PER_EVENT_UNITS",
    "ATTENTION_POLICY_SWITCH_COST_UNITS",
    "ATTENTION_POLICY_WEIGHT_TOTAL",
    "ATTENTION_POLICY_WEIGHTS",
    "ATTENTION_PROMPT_AUTHORITY_INTRO",
    "ATTENTION_PROMPT_FRAME_BYTES",
    "ATTENTION_PROMPT_FRAME_TEXT",
    "ATTENTION_PROMPT_SECTION_HEADERS",
    "AttentionCandidateDecision",
    "AttentionCandidateReason",
    "AttentionCompetitionOutcome",
    "AttentionCompetitionResult",
    "AttentionContinuityBaseline",
    "AttentionFallbackReason",
    "AttentionPromptDecision",
    "AttentionPromptReason",
    "AttentionPromptSelection",
    "AttentionSignalDimension",
    "AttentionUnfinishedOverflowError",
    "compete_attention",
    "next_habituation_units",
    "next_inhibition_units",
    "next_streak_counts",
    "select_attention_prompt",
]
