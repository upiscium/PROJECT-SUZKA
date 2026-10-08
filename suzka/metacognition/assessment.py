"""Deterministic U3 assessment over sealed, event-scoped observations.

This module is deliberately ephemeral: it consumes a validated observation and
returns the existing immutable assessment contract.  It has no history,
configuration, model, or Attention mutation authority.  The quality score is a
fixed current-resource-pressure heuristic, not an empirical measure of
capability or bias.
"""

from __future__ import annotations

from typing import Final

from suzka.attention.common import ATTENTION_MAX_FOCUS
from suzka.metacognition.contracts import (
    METACOGNITION_MAX_ASSESSMENT_BYTES,
    METACOGNITION_MAX_REASON_CODES,
    EpistemicBoundary,
    EvidenceCondition,
    MetacognitiveAssessment,
    MetacognitiveEvidenceWitness,
    MetacognitiveReasonCode,
)
from suzka.metacognition.evidence import (
    METACOGNITION_UNITS_SCALE,
    MetacognitionObservation,
)


METACOGNITION_ASSESSMENT_SCALE: Final[int] = METACOGNITION_UNITS_SCALE
# V1 tuple order: cognitive load, attention saturation, emotion influence.
METACOGNITION_QUALITY_WEIGHTS: Final[tuple[int, int, int]] = (400, 350, 250)
METACOGNITION_LOW_CONFIDENCE_THRESHOLD_UNITS: Final[int] = 500_000

_QUALITY_DENOMINATOR: Final[int] = sum(METACOGNITION_QUALITY_WEIGHTS)
_CONFIDENCE_DENOMINATOR: Final[int] = METACOGNITION_ASSESSMENT_SCALE**2


def _checked_units(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= METACOGNITION_ASSESSMENT_SCALE:
        raise ValueError(f"{name} must be an exact fixed-point integer in [0, 1000000]")
    return value


def _fixed_fraction(units: int | None) -> float | None:
    return None if units is None else units / METACOGNITION_ASSESSMENT_SCALE


def _quality_units(
    load_units: int | None,
    saturation_units: int | None,
    emotion_units: int | None,
) -> int | None:
    if load_units is None or saturation_units is None or emotion_units is None:
        return None
    load_weight, saturation_weight, emotion_weight = METACOGNITION_QUALITY_WEIGHTS
    weighted_pressure = (
        load_weight * load_units
        + saturation_weight * saturation_units
        + emotion_weight * emotion_units
    )
    # The weighted mean is floored in integer space before subtraction.  This
    # exact rational policy avoids float-order dependence and is clamped even
    # though validated input units and coefficients already imply [0, scale].
    pressure_units = weighted_pressure // _QUALITY_DENOMINATOR
    return max(
        0,
        min(METACOGNITION_ASSESSMENT_SCALE, METACOGNITION_ASSESSMENT_SCALE - pressure_units),
    )


def _supporting_ceilings(
    evidence: tuple[MetacognitiveEvidenceWitness, ...],
) -> tuple[float, ...]:
    ceilings: list[float] = []
    for witness in evidence:
        ceiling = witness.confidence_ceiling
        if witness.condition is EvidenceCondition.SUPPORTING and ceiling is not None:
            ceilings.append(ceiling)
    return tuple(ceilings)


def _reason_codes(
    *,
    observation: MetacognitionObservation,
    all_unknown: bool,
    contradictory: bool,
    has_supporting_ceiling: bool,
    coverage_units: int | None,
    load_units: int | None,
    saturation_units: int | None,
    emotion_units: int | None,
    quality_units: int | None,
    confidence_units: int | None,
) -> tuple[MetacognitiveReasonCode, ...]:
    reasons = set(observation.reason_codes)

    # These codes are derived from the resulting assessment, rather than from
    # caller labels.  Conflict codes are retained only when the typed source
    # witnesses establish the conflict boundary.
    if not contradictory:
        reasons.discard(MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE)
        reasons.discard(MetacognitiveReasonCode.CONFLICTING_SOURCES)
    else:
        reasons.add(MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE)
        reasons.add(MetacognitiveReasonCode.CONFLICTING_SOURCES)

    for code, units in (
        (MetacognitiveReasonCode.LOAD_UNOBSERVED, load_units),
        (MetacognitiveReasonCode.SATURATION_UNOBSERVED, saturation_units),
        (MetacognitiveReasonCode.EMOTION_INFLUENCE_UNOBSERVED, emotion_units),
        (MetacognitiveReasonCode.QUALITY_UNOBSERVED, quality_units),
    ):
        if units is not None:
            reasons.discard(code)

    if all_unknown or not has_supporting_ceiling:
        reasons.add(MetacognitiveReasonCode.MISSING_EVIDENCE)
    if load_units is None:
        reasons.add(MetacognitiveReasonCode.LOAD_UNOBSERVED)
    if saturation_units is None:
        reasons.add(MetacognitiveReasonCode.SATURATION_UNOBSERVED)
    if emotion_units is None:
        reasons.add(MetacognitiveReasonCode.EMOTION_INFLUENCE_UNOBSERVED)
    if quality_units is None:
        reasons.add(MetacognitiveReasonCode.QUALITY_UNOBSERVED)
    if coverage_units is not None and coverage_units < METACOGNITION_ASSESSMENT_SCALE:
        reasons.add(MetacognitiveReasonCode.PARTIAL_COVERAGE)
        reasons.add(MetacognitiveReasonCode.INSUFFICIENT_EVIDENCE)
    if (
        confidence_units is not None
        and confidence_units < METACOGNITION_LOW_CONFIDENCE_THRESHOLD_UNITS
    ):
        reasons.add(MetacognitiveReasonCode.LOW_CONFIDENCE)

    # A supplied Belief subset and event-scoped resources cannot establish
    # completeness of relevant knowledge.  Keep this boundary explicit even
    # when the measured supplied-link coverage is 100%.
    reasons.add(MetacognitiveReasonCode.LIMITED_PROVENANCE)
    reasons.add(MetacognitiveReasonCode.EVIDENCE_BOUNDARY)

    ordered = tuple(sorted(reasons, key=lambda code: code.value))
    if len(ordered) > METACOGNITION_MAX_REASON_CODES:
        raise ValueError("derived metacognitive reason codes exceed their closed bound")
    return ordered


def assess_metacognition(
    observation: MetacognitionObservation,
) -> MetacognitiveAssessment:
    """Purely assess one exact sealed observation using fixed integer policy.

    No numeric field can be overridden by the caller.  ``validated_copy`` is
    required to compare the observation's declared checksum with its current
    fields; this function never repairs a modified packet.
    """

    if type(observation) is not MetacognitionObservation:
        raise TypeError("observation must be an exact MetacognitionObservation")
    checked = observation.validated_copy()
    if type(checked) is not MetacognitionObservation:
        raise TypeError("validated observation must remain an exact MetacognitionObservation")

    if type(checked.evidence) is not tuple or any(
        type(witness) is not MetacognitiveEvidenceWitness
        for witness in checked.evidence
    ):
        raise TypeError("observation evidence must contain exact typed witnesses")
    evidence = tuple(
        sorted(checked.evidence, key=lambda witness: witness.witness_digest)
    )

    load_units = _checked_units(checked.cognitive_load_units, "cognitive_load_units")
    saturation_units = _checked_units(
        checked.attention_saturation_units, "attention_saturation_units"
    )
    emotion_units = _checked_units(
        checked.emotion_influence_units, "emotion_influence_units"
    )
    coverage_units = _checked_units(
        checked.belief_coverage_units, "belief_coverage_units"
    )
    ceiling_units = _checked_units(
        checked.belief_confidence_ceiling_units,
        "belief_confidence_ceiling_units",
    )
    if (
        type(checked.focus_count) is not int
        or not 0 <= checked.focus_count <= ATTENTION_MAX_FOCUS
    ):
        raise ValueError("focus_count must be a bounded exact integer")
    if type(checked.contradictory) is not bool:
        raise TypeError("contradictory must be an exact bool")

    has_observed_evidence = any(
        witness.condition is not EvidenceCondition.UNKNOWN for witness in evidence
    )
    contradictory = checked.contradictory or any(
        witness.condition is EvidenceCondition.CONTRADICTORY
        for witness in evidence
    )
    all_unknown = not has_observed_evidence
    supporting_ceilings = _supporting_ceilings(evidence)
    has_supporting_ceiling = bool(supporting_ceilings)
    raw_ceiling = min(supporting_ceilings) if supporting_ceilings else None

    if all_unknown:
        boundary = EpistemicBoundary.UNKNOWN
        result_load_units = None
        result_saturation_units = None
        result_emotion_units = None
        result_coverage_units = None
        result_quality_units = None
        confidence_units = None
        confidence = None
    else:
        boundary = (
            EpistemicBoundary.CONTRADICTORY
            if contradictory
            else EpistemicBoundary.UNCERTAIN
        )
        result_load_units = load_units
        result_saturation_units = saturation_units
        result_emotion_units = emotion_units
        result_quality_units = _quality_units(
            load_units,
            saturation_units,
            emotion_units,
        )

        coverage_is_observed = (
            coverage_units is not None and checked.focus_count > 0
        )
        result_coverage_units = (
            0
            if contradictory and coverage_is_observed
            else coverage_units
            if coverage_is_observed
            else None
        )

        confidence_ready = (
            has_supporting_ceiling
            and ceiling_units is not None
            and coverage_is_observed
            and result_quality_units is not None
        )
        if contradictory and confidence_ready:
            confidence_units = 0
            confidence = 0.0
        elif confidence_ready:
            assert ceiling_units is not None
            assert result_coverage_units is not None
            assert result_quality_units is not None
            confidence_units = (
                ceiling_units * result_coverage_units * result_quality_units
            ) // _CONFIDENCE_DENOMINATOR
            confidence = confidence_units / METACOGNITION_ASSESSMENT_SCALE
            # Binary64 division at the unit boundary must not lift confidence
            # above the raw typed R12 ceiling retained by a supporting witness.
            assert raw_ceiling is not None
            confidence = min(confidence, raw_ceiling)
        else:
            confidence_units = None
            confidence = None

    reason_codes = _reason_codes(
        observation=checked,
        all_unknown=all_unknown,
        contradictory=contradictory,
        has_supporting_ceiling=has_supporting_ceiling,
        coverage_units=(
            coverage_units
            if coverage_units is not None and checked.focus_count > 0
            else None
        ),
        load_units=result_load_units,
        saturation_units=result_saturation_units,
        emotion_units=result_emotion_units,
        quality_units=result_quality_units,
        confidence_units=confidence_units,
    )

    assessment = MetacognitiveAssessment(
        event=checked.event,
        focus_witness=checked.focus_witness,
        evidence=evidence,
        evidence_sufficiency=_fixed_fraction(result_coverage_units),
        epistemic_boundary=boundary,
        confidence=confidence,
        cognitive_load=_fixed_fraction(result_load_units),
        attention_saturation=_fixed_fraction(result_saturation_units),
        emotion_influence=_fixed_fraction(result_emotion_units),
        cognitive_quality=_fixed_fraction(result_quality_units),
        reason_codes=reason_codes,
    )
    encoded = assessment.canonical_bytes()
    if len(encoded) > METACOGNITION_MAX_ASSESSMENT_BYTES:
        raise ValueError("metacognitive assessment exceeds its schema envelope")
    return assessment


__all__ = [
    "METACOGNITION_ASSESSMENT_SCALE",
    "METACOGNITION_LOW_CONFIDENCE_THRESHOLD_UNITS",
    "METACOGNITION_QUALITY_WEIGHTS",
    "assess_metacognition",
]
