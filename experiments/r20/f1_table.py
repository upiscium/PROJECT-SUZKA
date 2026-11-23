"""R20 IP-22b: the F1 expected-value table (IP-22 D1 r3 sec. 5, HD-22-6 (a)).

Pure, no I/O.  The table is the one the merged SuccessorPolicyV2 code
produces (``interpretation_from_r15`` plus ``evaluate``; #374 D3 sec. 3), not
#374's earlier 2026-11-08 class definition.  The per-pair class is derived
from the evaluated row; a mismatch with this table is
``GATE_CONFORMANCE_DEFECT``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from experiments.r20.ablation import DefectClass
from suzka.r16.successor_policy import (
    PolicyOutputKind,
    SuccessorPolicyInputV2,
    SuccessorRow,
    evaluate,
    interpretation_from_r15,
)

#: R15 InterpretationStatus (stage S2) -> (row, output) of the merged code.
F1_EXPECTED_TABLE: Final[dict[str, tuple[SuccessorRow, PolicyOutputKind]]] = {
    # WRITE intent is absent in F1 (r18 is off), so row 4 is Plan B only as a policy output.
    "PRESENT": (SuccessorRow.ROW_4, PolicyOutputKind.PLAN_B),
    "CONTESTED": (SuccessorRow.ROW_4B, PolicyOutputKind.PLAN_A),
    "UNKNOWN": (SuccessorRow.ROW_4B, PolicyOutputKind.PLAN_A),
    "CONTRADICTORY": (SuccessorRow.ROW_2B, PolicyOutputKind.HOLD),
}

#: HD-22-9 (i): 8 treatment pairs + 4 control pairs = 12 pairs, 24 subjects.
F1_TREATMENT_PAIRS: Final = 8
F1_CONTROL_PAIRS: Final = 4  # 2 NEUTRAL/NEUTRAL + 2 UNCERTAINTY/UNCERTAINTY
F1_TOTAL_PAIRS: Final = F1_TREATMENT_PAIRS + F1_CONTROL_PAIRS

F1_PREREGISTRATION_TEXT: Final = (
    "F1 expected-value table (HD-22-6 (a), the merged-code table): at stage S2 "
    "R15 status PRESENT maps to successor row 4 (Plan B), CONTESTED and UNKNOWN "
    "map to row 4b (Plan A), and CONTRADICTORY maps to row 2b (HOLD). This "
    "table follows the merged SuccessorPolicyV2 code and #374 D3 sec. 3, not "
    "the earlier #374 D3-4 class definition: UNKNOWN is not in the baseline "
    "class, and gate-relevant discordance is row 4 against row 4b / 2b. A "
    "mismatch between this table and the evaluated row is a "
    "GATE_CONFORMANCE_DEFECT. F1 uses 8 treatment pairs plus 4 control pairs "
    "(2 NEUTRAL/NEUTRAL, 2 UNCERTAINTY/UNCERTAINTY) = 12 pairs (HD-22-9 (i)). "
    "Limitation wording (HD-22-8 (a)): R16 consumes R15's status class through "
    "a code-owned table; the Feedback ablation fixes interpretation to SETTLED / S2."
)


@dataclass(frozen=True)
class F1StatusResult:
    status: str
    expected_row: SuccessorRow
    expected_output: PolicyOutputKind
    evaluated_row: SuccessorRow
    evaluated_output: PolicyOutputKind
    defect: DefectClass | None


def evaluate_f1_status(payload: bytes, status: str) -> F1StatusResult:
    """Evaluate one S2 R15 status on a pair's sealed non-S input.

    ``payload`` is the sealed input with Feedback / ledger as the pair has
    them; only ``interpretation`` and its stage are replaced, through the
    merged R15 map.  An unknown status is refused (``ValueError``).
    """

    if type(status) is not str or status not in F1_EXPECTED_TABLE:
        raise ValueError("unknown F1 R15 status")
    interpretation, stage = interpretation_from_r15("EPISODE", "S2", status)
    base = SuccessorPolicyInputV2.from_bytes(payload)
    result = evaluate(replace(base, interpretation=interpretation, interpretation_stage=stage))
    expected_row, expected_output = F1_EXPECTED_TABLE[status]
    mismatch = (result.row, result.output) != (expected_row, expected_output)
    return F1StatusResult(
        status=status,
        expected_row=expected_row,
        expected_output=expected_output,
        evaluated_row=result.row,
        evaluated_output=result.output,
        defect=DefectClass.GATE_CONFORMANCE_DEFECT if mismatch else None,
    )


def evaluate_f1_table(payload: bytes) -> tuple[F1StatusResult, ...]:
    return tuple(evaluate_f1_status(payload, status) for status in F1_EXPECTED_TABLE)
