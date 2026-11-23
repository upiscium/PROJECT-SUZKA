"""R20 IP-22b: the pure Feedback-ablation evaluator and the F1 table (D1 sec. 5)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.r20 import ablation, f1_table
from experiments.r20.ablation import (
    AblationLabel,
    DefectClass,
    ablate_feedback,
    classify_difference,
    interpretation_variant,
    interpretation_variants,
    replay_defect,
)
from experiments.r20.f1_table import (
    F1_EXPECTED_TABLE,
    F1_PREREGISTRATION_TEXT,
    evaluate_f1_status,
    evaluate_f1_table,
)
from suzka.r16.successor_policy import (
    NO_PREDECESSOR_HORIZON_KIND,
    CommitmentState,
    FeedbackState,
    Interpretation,
    InterpretationStage,
    LedgerState,
    LineageRef,
    LineageState,
    PolicyOutputKind,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorRow,
    evaluate,
)

D = "a" * 64
SOURCES = tuple(PolicyRef(k, f"{k}-1", 1, D) for k in ("context", "experience", "goal", "value"))
LINEAGE = LineageRef("lineage-1", "baseline-1", D, None, "b" * 64)
FEEDBACK_REF = PolicyRef("feedback", "feedback-1", 1, D)


def _base(**changes: object) -> SuccessorPolicyInputV2:
    """Row 4 family: initial, ACTIVE Commitment, nothing executed."""

    value = SuccessorPolicyInputV2(
        required_source=RequiredSourceState.QUALIFIED,
        source_refs=SOURCES,
        predecessor=PredecessorState.ABSENT_INITIAL,
        predecessor_ref=PolicyRef(NO_PREDECESSOR_HORIZON_KIND, "horizon-9", 9, D),
        lineage=LineageState.CONSISTENT,
        lineage_ref=LINEAGE,
        commitment=CommitmentState.ACTIVE,
        commitment_ref=PolicyRef("commitment", "commitment-1", 1, D),
        commitment_deadline_utc="2026-11-21T00:00:00Z",
        ledger=LedgerState.NONE,
        ledger_refs=(),
        inverse_authorization=None,
        feedback=FeedbackState.ABSENT,
        feedback_ref=None,
        verified_outcome_ref=None,
        interpretation=Interpretation.SETTLED,
        interpretation_stage=InterpretationStage.S2,
        as_of_r07_sequence=9,
        as_of_snapshot_hash="c" * 64,
        event_id="event-10",
        event_sequence=10,
        now_utc="2026-11-14T00:00:00Z",
    )
    return replace(value, **changes)  # type: ignore[arg-type]


def _s0(**changes: object) -> SuccessorPolicyInputV2:
    """Row 5: verified authorized effect, MATCH_SELF, SETTLED / S2."""

    values: dict[str, object] = dict(
        predecessor=PredecessorState.QUALIFIED,
        predecessor_ref=PolicyRef("decision", "decision-1", 1, D),
        ledger=LedgerState.ACTIVE_VERIFIED_AUTHORIZED,
        ledger_refs=(PolicyRef("ledger", "ledger-1", 1, D),),
        verified_outcome_ref=PolicyRef("outcome", "outcome-1", 1, D),
        feedback=FeedbackState.MATCH_SELF,
        feedback_ref=FEEDBACK_REF,
    )
    values.update(changes)
    return _base(**values)


# ------------------------------------------------------------ the pair


def test_s0_pair_is_row_5_against_row_6() -> None:
    value = _s0()
    result = ablate_feedback(value.canonical_bytes())
    assert result.label is AblationLabel.S0_ABLATION
    assert result.pair == (SuccessorRow.ROW_5, SuccessorRow.ROW_6)
    assert (result.factual_output, result.ablated_output) == (
        PolicyOutputKind.NO_OP,
        PolicyOutputKind.PLAN_A,
    )
    assert result.factual_write_eligible is False
    assert result.ablated_write_eligible is False
    assert result.s0_pair_defect() is None


def test_differ_proves_only_feedback_and_its_ref_changed() -> None:
    value = _s0()
    result = ablate_feedback(value.canonical_bytes())
    assert result.differing_fields == ("feedback", "feedback_ref")
    hypo = SuccessorPolicyInputV2.from_bytes(result.ablated_payload)
    assert hypo.feedback is FeedbackState.ABSENT and hypo.feedback_ref is None
    assert ablation.fields_differing(value, hypo) == ablation.FEEDBACK_FIELDS
    assert ablation.fields_differing(value, value) == frozenset()


def test_nulling_only_the_state_would_yield_row_0_not_row_6() -> None:
    """Negative test: the ref must be nulled too (an ABSENT state keeping a ref)."""

    wrong = replace(_s0(), feedback=FeedbackState.ABSENT)
    assert evaluate(wrong).row is SuccessorRow.ROW_0
    result = ablate_feedback(_s0().canonical_bytes())
    assert result.ablated_row is SuccessorRow.ROW_6
    assert result.ablated_row is not SuccessorRow.ROW_0


def test_non_row_5_pair_on_a_settled_input_is_a_gate_conformance_defect() -> None:
    # ledger/feedback arranged so the factual row is 6 (Feedback MISMATCH): pair (6, 6).
    value = _s0(feedback=FeedbackState.MISMATCH)
    result = ablate_feedback(value.canonical_bytes())
    assert result.is_s0_ablation
    assert result.pair == (SuccessorRow.ROW_6, SuccessorRow.ROW_6)
    assert result.s0_pair_defect() is DefectClass.GATE_CONFORMANCE_DEFECT


@pytest.mark.parametrize(
    ("interpretation", "stage"),
    [
        (Interpretation.UNCERTAIN, InterpretationStage.S2),
        (Interpretation.NOT_REVIEWED, InterpretationStage.NONE),
        (Interpretation.CONTRADICTORY, InterpretationStage.S2),
        (Interpretation.SETTLED, InterpretationStage.S1),
    ],
)
def test_only_settled_s2_is_an_s0_ablation(
    interpretation: Interpretation, stage: InterpretationStage
) -> None:
    result = ablate_feedback(
        _s0(interpretation=interpretation, interpretation_stage=stage).canonical_bytes()
    )
    assert result.label is AblationLabel.NOT_AN_S0_ABLATION
    assert result.s0_pair_defect() is None
    # The rows are still reported.
    assert result.factual_row in SuccessorRow


def test_uncertain_gives_5b_against_6_and_is_not_an_s0_ablation() -> None:
    result = ablate_feedback(_s0(interpretation=Interpretation.UNCERTAIN).canonical_bytes())
    assert result.pair == (SuccessorRow.ROW_5B, SuccessorRow.ROW_6)
    assert result.label is AblationLabel.NOT_AN_S0_ABLATION


def test_malformed_payload_is_refused() -> None:
    with pytest.raises(ValueError):
        ablate_feedback(b"{}")
    with pytest.raises(ValueError):
        ablate_feedback(_s0().canonical_bytes() + b" ")


# --------------------------------------------------- replay and labels


def test_replay_check_and_classification() -> None:
    payload = _s0().canonical_bytes()
    assert replay_defect(payload, SuccessorRow.ROW_5) is None
    assert replay_defect(payload, SuccessorRow.ROW_6) is DefectClass.ENGINEERING_DEFECT
    kwargs = {"stored_payload": payload}
    assert (
        classify_difference(stored_row=SuccessorRow.ROW_6, new_qualified_event=True, **kwargs)
        is DefectClass.ENGINEERING_DEFECT
    )
    assert (
        classify_difference(stored_row=SuccessorRow.ROW_5, new_qualified_event=True, **kwargs)
        is DefectClass.COGNITIVE_VARIATION
    )
    assert (
        classify_difference(stored_row=SuccessorRow.ROW_5, new_qualified_event=False, **kwargs)
        is DefectClass.UNDETERMINED
    )
    assert (
        classify_difference(stored_payload=None, stored_row=None, new_qualified_event=True)
        is DefectClass.UNDETERMINED
    )
    assert {c.value for c in DefectClass} == {
        "GATE_CONFORMANCE_DEFECT",
        "ENGINEERING_DEFECT",
        "COGNITIVE_VARIATION",
        "UNDETERMINED",
    }


# ------------------------------------------------ interpretation variants


def test_interpretation_variants_are_pinned_and_leave_feedback_alone() -> None:
    payload = _s0().canonical_bytes()
    rows = {(v.interpretation, v.stage): v.row for v in interpretation_variants(payload)}
    assert rows == {
        (Interpretation.NOT_REVIEWED, InterpretationStage.NONE): SuccessorRow.ROW_5,
        (Interpretation.SETTLED, InterpretationStage.S2): SuccessorRow.ROW_5,
        (Interpretation.UNCERTAIN, InterpretationStage.S2): SuccessorRow.ROW_5B,
        (Interpretation.CONTRADICTORY, InterpretationStage.S2): SuccessorRow.ROW_2B,
    }


def test_illegal_stage_interpretation_pair_is_refused() -> None:
    payload = _s0().canonical_bytes()
    with pytest.raises(ValueError):
        interpretation_variant(payload, Interpretation.CONTRADICTORY, InterpretationStage.S1)
    with pytest.raises(ValueError):
        interpretation_variant(payload, Interpretation.NOT_REVIEWED, InterpretationStage.S2)


# -------------------------------------------------------------- F1 table


def test_f1_table_is_the_merged_code_table() -> None:
    results = {r.status: r for r in evaluate_f1_table(_base().canonical_bytes())}
    assert {k: (r.evaluated_row, r.evaluated_output) for k, r in results.items()} == {
        "PRESENT": (SuccessorRow.ROW_4, PolicyOutputKind.PLAN_B),
        "CONTESTED": (SuccessorRow.ROW_4B, PolicyOutputKind.PLAN_A),
        "UNKNOWN": (SuccessorRow.ROW_4B, PolicyOutputKind.PLAN_A),
        "CONTRADICTORY": (SuccessorRow.ROW_2B, PolicyOutputKind.HOLD),
    }
    assert all(r.defect is None for r in results.values())
    assert {k: v[0] for k, v in F1_EXPECTED_TABLE.items()} == {
        k: r.evaluated_row for k, r in results.items()
    }


def test_f1_mismatch_is_a_gate_conformance_defect() -> None:
    # A row-3 input (no Commitment) does not produce row 4 for PRESENT.
    payload = _base(
        commitment=CommitmentState.ABSENT, commitment_ref=None, commitment_deadline_utc=None
    ).canonical_bytes()
    result = evaluate_f1_status(payload, "PRESENT")
    assert result.evaluated_row is SuccessorRow.ROW_3
    assert result.defect is DefectClass.GATE_CONFORMANCE_DEFECT


def test_f1_unknown_status_refused() -> None:
    with pytest.raises(ValueError):
        evaluate_f1_status(_base().canonical_bytes(), "NOT_A_STATUS")


def test_f1_preregistration_names_the_table_and_decisions() -> None:
    text = F1_PREREGISTRATION_TEXT
    for needle in (
        "row 4 (Plan B)",
        "row 4b (Plan A)",
        "row 2b (HOLD)",
        "merged",
        "GATE_CONFORMANCE_DEFECT",
        "12 pairs",
        "SETTLED / S2",
    ):
        assert needle in text
    assert f1_table.F1_TOTAL_PAIRS == 12


# ----------------------------------------------------- purity / determinism

_ALLOWED_TOP = {
    "dataclasses",
    "enum",
    "__future__",
    "suzka.r16.successor_policy",
}


def test_module_imports_only_the_pure_policy_surface() -> None:
    tree = ast.parse(Path(ablation.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module or "")
    assert imported <= _ALLOWED_TOP


def test_importing_the_module_loads_no_store_runtime_or_api() -> None:
    code = (
        "import sys, experiments.r20.ablation;"
        "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('suzka'))))"
    )
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    allowed = {
        "suzka",
        "suzka.identifiers",
        "suzka.limits",
        "suzka.r16",
        "suzka.r16.contracts",
        "suzka.r16.history_codec",
        "suzka.r16.successor_policy",
    }
    assert set(out) <= allowed


def test_deterministic_and_stores_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _s0().canonical_bytes()
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "f").write_bytes(b"x")

    def snapshot() -> dict[str, bytes]:
        return {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    before = snapshot()

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the evaluator must not open files")

    monkeypatch.setattr("builtins.open", refuse)
    first = ablate_feedback(payload)
    second = ablate_feedback(payload)
    monkeypatch.undo()
    assert first == second
    assert first.ablated_payload == second.ablated_payload
    assert snapshot() == before
    # the factual input bytes are not modified (immutability of the caller's value)
    assert payload == _s0().canonical_bytes()
