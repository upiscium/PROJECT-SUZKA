"""R20 IP-22b: the pure Feedback-ablation evaluator (IP-22 D1 r3 sec. 5).

Pure functions over bytes and enums.  This module opens no store, takes no
runtime, no clock, no model and no file, and it never passes through a source
owner's admission: its results are values for an evidence manifest only, never
an event, Experience or Decision.  The caller extracts the sealed
``policy_input_bytes`` of a committed Decision and passes them in.

It imports only ``suzka.r16.successor_policy`` (which itself rests on
``suzka.limits`` / ``suzka.r16.contracts`` / ``suzka.r16.history_codec``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum

from suzka.r16.successor_policy import (
    FeedbackState,
    Interpretation,
    InterpretationStage,
    PolicyOutputKind,
    SuccessorPolicyInputV2,
    SuccessorRow,
    cross_reference_conflicts,
    evaluate,
)


class DefectClass(str, Enum):
    """The R20 reading of a difference (D1 sec. 5 / 6, #370 D1, #374 D3)."""

    GATE_CONFORMANCE_DEFECT = "GATE_CONFORMANCE_DEFECT"
    ENGINEERING_DEFECT = "ENGINEERING_DEFECT"
    COGNITIVE_VARIATION = "COGNITIVE_VARIATION"
    UNDETERMINED = "UNDETERMINED"


class AblationLabel(str, Enum):
    S0_ABLATION = "S0_ABLATION"
    NOT_AN_S0_ABLATION = "NOT_AN_S0_ABLATION"


#: The S0 chain's expected pair (D1 sec. 5 rule 2): factual row 5 against row 6.
S0_EXPECTED_PAIR: tuple[SuccessorRow, SuccessorRow] = (SuccessorRow.ROW_5, SuccessorRow.ROW_6)
#: The only two input fields the Feedback ablation may change (rule 1).
FEEDBACK_FIELDS: frozenset[str] = frozenset({"feedback", "feedback_ref"})


def fields_differing(a: SuccessorPolicyInputV2, b: SuccessorPolicyInputV2) -> frozenset[str]:
    """Names of the sealed-input fields whose values differ."""

    if type(a) is not SuccessorPolicyInputV2 or type(b) is not SuccessorPolicyInputV2:
        raise TypeError("fields_differing needs exact SuccessorPolicyInputV2 values")
    return frozenset(f.name for f in fields(a) if getattr(a, f.name) != getattr(b, f.name))


@dataclass(frozen=True)
class AblationResult:
    label: AblationLabel
    factual_row: SuccessorRow
    factual_output: PolicyOutputKind
    ablated_row: SuccessorRow
    ablated_output: PolicyOutputKind
    differing_fields: tuple[str, ...]
    factual_write_eligible: bool
    ablated_write_eligible: bool
    ablated_payload: bytes

    @property
    def pair(self) -> tuple[SuccessorRow, SuccessorRow]:
        return (self.factual_row, self.ablated_row)

    @property
    def is_s0_ablation(self) -> bool:
        return self.label is AblationLabel.S0_ABLATION

    def s0_pair_defect(self) -> DefectClass | None:
        """``GATE_CONFORMANCE_DEFECT`` unless an S0 claim yields ``(5, 6)``.

        A ``NOT_AN_S0_ABLATION`` result is not reported as the Feedback
        ablation at all, so it has no S0 verdict (``None``).
        """

        if not self.is_s0_ablation:
            return None
        if self.pair != S0_EXPECTED_PAIR:
            return DefectClass.GATE_CONFORMANCE_DEFECT
        return None


def ablate_feedback(payload: bytes) -> AblationResult:
    """Evaluate a sealed input against its Feedback-ABSENT counterfactual."""

    factual = SuccessorPolicyInputV2.from_bytes(payload)
    # Rule 1: the Feedback state AND its own reference (an ABSENT state that
    # keeps a ref is a row-0 conflict, successor_policy "absent_feedback_has_ref").
    hypo = replace(factual, feedback=FeedbackState.ABSENT, feedback_ref=None)
    differing = fields_differing(factual, hypo)
    if not differing <= FEEDBACK_FIELDS:
        raise ValueError("the ablation changed a field other than Feedback")
    if cross_reference_conflicts(hypo):
        raise ValueError("the ablated input is not a valid closed input")
    factual_result = evaluate(factual)
    ablated_result = evaluate(hypo)
    # Rule 6: an S0 claim needs SETTLED / S2.
    s0 = (
        factual.interpretation is Interpretation.SETTLED
        and factual.interpretation_stage is InterpretationStage.S2
    )
    return AblationResult(
        label=AblationLabel.S0_ABLATION if s0 else AblationLabel.NOT_AN_S0_ABLATION,
        factual_row=factual_result.row,
        factual_output=factual_result.output,
        ablated_row=ablated_result.row,
        ablated_output=ablated_result.output,
        differing_fields=tuple(sorted(differing)),
        factual_write_eligible=factual_result.write_eligible,
        ablated_write_eligible=ablated_result.write_eligible,
        ablated_payload=hypo.canonical_bytes(),
    )


def replay_defect(payload: bytes, stored_row: SuccessorRow) -> DefectClass | None:
    """Rule 3: an identical sealed input must replay to the stored row."""

    if type(stored_row) is not SuccessorRow:
        raise TypeError("stored_row must be an exact SuccessorRow")
    if evaluate(SuccessorPolicyInputV2.from_bytes(payload)).row is not stored_row:
        return DefectClass.ENGINEERING_DEFECT
    return None


def classify_difference(
    *,
    stored_payload: bytes | None,
    stored_row: SuccessorRow | None,
    new_qualified_event: bool,
) -> DefectClass:
    """Only the identical-sealed-input replay decides; else ``UNDETERMINED``.

    Replay differs: ``ENGINEERING_DEFECT``.  Replay equal and the difference
    is a new qualified event: ``COGNITIVE_VARIATION``.  Anything else
    (missing evidence, no new qualified event): ``UNDETERMINED``.
    """

    if stored_payload is None or stored_row is None:
        return DefectClass.UNDETERMINED
    if replay_defect(stored_payload, stored_row) is not None:
        return DefectClass.ENGINEERING_DEFECT
    if new_qualified_event:
        return DefectClass.COGNITIVE_VARIATION
    return DefectClass.UNDETERMINED


@dataclass(frozen=True)
class InterpretationVariant:
    interpretation: Interpretation
    stage: InterpretationStage
    row: SuccessorRow
    output: PolicyOutputKind


def interpretation_variant(
    payload: bytes, interpretation: Interpretation, stage: InterpretationStage
) -> InterpretationVariant:
    """Rule 5: replace only ``interpretation`` / its stage; Feedback untouched.

    An illegal ``(stage, interpretation)`` pair is refused by the input's own
    construction checks (``ValueError``), never silently evaluated.
    """

    factual = SuccessorPolicyInputV2.from_bytes(payload)
    variant = replace(factual, interpretation=interpretation, interpretation_stage=stage)
    result = evaluate(variant)
    return InterpretationVariant(interpretation, stage, result.row, result.output)


#: The legal variants of the R15 limitation analysis (sec. 12, A-4).
LIMITATION_VARIANTS: tuple[tuple[Interpretation, InterpretationStage], ...] = (
    (Interpretation.NOT_REVIEWED, InterpretationStage.NONE),
    (Interpretation.SETTLED, InterpretationStage.S2),
    (Interpretation.UNCERTAIN, InterpretationStage.S2),
    (Interpretation.CONTRADICTORY, InterpretationStage.S2),
)


def interpretation_variants(payload: bytes) -> tuple[InterpretationVariant, ...]:
    return tuple(interpretation_variant(payload, i, s) for i, s in LIMITATION_VARIANTS)
