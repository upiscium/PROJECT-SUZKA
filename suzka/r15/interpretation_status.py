"""R15 IP-8a: ``interpretation_status_v2`` — the closed status table (#374 F1 D2 §2).

A pure function from the typed self-assessment the subject reported (S1/S2
candidate) to the Interpretation shape: status, which witness supports or
opposes, and the confidence derived in code.  It reads no source and grants
no authority; the source gate qualifies the inputs first.

Rows (D2 §2, D3 §1).  PRESENT needs uncertainty LOW/MEDIUM in both stages.

| stage/relation | assessment | uncertainty | status | support | counter | experience |
| S1 / KEEPS | LIKELY_ALIGNED | LOW/MEDIUM | PRESENT | first | () | first |
| S1 / KEEPS | LIKELY_ALIGNED | HIGH | UNKNOWN | () | () | first |
| S1 / KEEPS | MAY_HAVE_MISREAD/UNKNOWN | any | UNKNOWN | () | () | first |
| S2 / ADOPTS | MAY_HAVE_MISREAD | LOW/MEDIUM | PRESENT | later | () | later |
| S2 / ADOPTS | MAY_HAVE_MISREAD | HIGH | CONTESTED | later | first | later |
| S2 / ADOPTS | LIKELY_ALIGNED/UNKNOWN | any | CONTESTED | later | first | later |
| S2 / KEEPS | LIKELY_ALIGNED | any | CONTRADICTORY | first | later | first |
| S2 / KEEPS | MAY_HAVE_MISREAD/UNKNOWN | any | CONTESTED | first | later | first |
| S2 / UNKNOWN | any | any | UNKNOWN | () | () | later |
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from suzka.r15.common import InterpretationStatus, RevisionReason


class Assessment(str, Enum):
    LIKELY_ALIGNED = "LIKELY_ALIGNED"
    MAY_HAVE_MISREAD = "MAY_HAVE_MISREAD"
    UNKNOWN = "UNKNOWN"


class Uncertainty(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ReferentRelation(str, Enum):
    """How the reported referent relates to the original choice (total for S2)."""

    KEEPS_ORIGINAL = "KEEPS_ORIGINAL"
    ADOPTS_REPORT = "ADOPTS_REPORT"
    UNKNOWN = "UNKNOWN"


class Witness(str, Enum):
    """Which observed witness a row role refers to."""

    FIRST = "FIRST"  # the original first-person answer E
    LATER = "LATER"  # the later user-reported correction E2


ESTIMATE: Final = 500_000
CONTESTED_CONFIDENCE: Final = 250_000
_CONFIDENCE: Final = {Uncertainty.LOW: 750_000, Uncertainty.MEDIUM: 500_000}


@dataclass(frozen=True, slots=True)
class StatusRow:
    status: InterpretationStatus
    experience: Witness
    support: tuple[Witness, ...]
    counter: tuple[Witness, ...]
    estimate: int | None
    confidence: int | None
    reason: RevisionReason


def interpretation_status_v2(
    stage: str,
    relation: ReferentRelation,
    assessment: Assessment,
    uncertainty: Uncertainty,
) -> StatusRow:
    if (
        type(relation) is not ReferentRelation
        or type(assessment) is not Assessment
        or type(uncertainty) is not Uncertainty
    ):
        raise TypeError("status inputs must be exact closed values")
    if stage == "S1":
        if relation is not ReferentRelation.KEEPS_ORIGINAL:
            raise ValueError("S1 can only keep the original referent")
        if assessment is Assessment.LIKELY_ALIGNED and uncertainty in _CONFIDENCE:
            return _present(Witness.FIRST, uncertainty, RevisionReason.INITIAL_INTERPRETATION)
        return _unknown(Witness.FIRST, RevisionReason.INITIAL_INTERPRETATION)
    if stage != "S2":
        raise ValueError("stage must be S1 or S2")
    if relation is ReferentRelation.UNKNOWN:
        return _unknown(Witness.LATER, RevisionReason.CONTRARY_EVIDENCE)
    if relation is ReferentRelation.ADOPTS_REPORT:
        reason = RevisionReason.QUALIFIED_REINTERPRETATION
        if assessment is Assessment.MAY_HAVE_MISREAD and uncertainty in _CONFIDENCE:
            return _present(Witness.LATER, uncertainty, reason)
        return _contested(Witness.LATER, Witness.FIRST, reason)
    reason = RevisionReason.CONTRARY_EVIDENCE
    if assessment is Assessment.LIKELY_ALIGNED:
        return StatusRow(
            InterpretationStatus.CONTRADICTORY,
            Witness.FIRST,
            (Witness.FIRST,),
            (Witness.LATER,),
            None,
            None,
            reason,
        )
    return _contested(Witness.FIRST, Witness.LATER, reason)


def _present(witness: Witness, uncertainty: Uncertainty, reason: RevisionReason) -> StatusRow:
    return StatusRow(
        InterpretationStatus.PRESENT,
        witness,
        (witness,),
        (),
        ESTIMATE,
        _CONFIDENCE[uncertainty],
        reason,
    )


def _contested(support: Witness, counter: Witness, reason: RevisionReason) -> StatusRow:
    return StatusRow(
        InterpretationStatus.CONTESTED,
        support,
        (support,),
        (counter,),
        ESTIMATE,
        CONTESTED_CONFIDENCE,
        reason,
    )


def _unknown(experience: Witness, reason: RevisionReason) -> StatusRow:
    return StatusRow(InterpretationStatus.UNKNOWN, experience, (), (), None, None, reason)


__all__ = [
    "Assessment",
    "ReferentRelation",
    "StatusRow",
    "Uncertainty",
    "Witness",
    "interpretation_status_v2",
]
