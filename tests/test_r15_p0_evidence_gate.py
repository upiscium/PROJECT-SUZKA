"""P0 evaluator-only source selection and fail-closed claims; NO R15 admission."""

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from experiments.r15_p0.evidence_gate import (
    GateStatus,
    ModelTriplet,
    SelectionStatus,
    UNKNOWN_TRIPLET,
    gate_model_claim,
    model_question,
    parse_triplet,
    select_evidence,
)
from experiments.r15_p0.pilot import (
    Case,
    Choice,
    Relation,
    run_case,
    seal_case,
)


def fixture(case: Case, *, original_response: str = "案Aを暫定的に選びました。"):
    temp = TemporaryDirectory(prefix="p0-evidence-gate-")
    root = Path(temp.name)
    manifest_file = root / "manifest.json"
    capture_file = root / "capture.json"
    manifest = seal_case(case, manifest_file)
    index = 0

    def transport(_message: str) -> dict[str, str]:
        nonlocal index
        index += 1
        return {
            "episode_id": f"synthetic-episode-{index}",
            "response": original_response if index == 2 else "テスト用可視回答",
        }

    capture = run_case(manifest_file, capture_file, transport)
    return temp, manifest, capture


def triplet(prior: str, target: str, relation: str) -> str:
    return (
        f"original_choice: {prior}\n"
        f"revealed_target: {target}\n"
        f"relation: {relation}"
    )


def test_complete_exact_e_and_e2_selection_does_not_include_unneeded_outputs() -> None:
    temp, manifest, capture = fixture(Case("budget-match", "B", "forced", 0))
    with temp:
        selected = select_evidence(manifest, capture)
        assert selected.status is SelectionStatus.INCLUDED
        assert selected.packet is not None
        sources = selected.packet.source_value()["sources"]
        assert [x["step"] for x in sources] == ["setup", "E", "E2"]
        assert sources[1]["user_input"] == capture["turns"][1]["request"]
        assert sources[1]["visible_answer"] == capture["turns"][1]["response"]
        assert sources[2]["user_input"] == capture["turns"][2]["request"]
        assert sources[2]["visible_answer"] is None
        assert sources[0]["visible_answer"] is None
        assert selected.required_bytes == selected.packet.source_bytes
        assert len(model_question(selected).encode("utf-8")) > selected.required_bytes
        assert "P0_BENCHMARK_OBSERVATION_ONLY" in model_question(selected)


def test_budget_fail_closed_no_truncation_or_model_call() -> None:
    temp, manifest, capture = fixture(Case("budget-reject", "A", "forced", 1))
    with temp:
        selected = select_evidence(manifest, capture, budget_bytes=256)
        assert selected.status is SelectionStatus.OVER_BUDGET
        assert selected.packet is None
        assert selected.required_bytes > 256
        with pytest.raises(ValueError, match="unavailable source"):
            model_question(selected)
        result = gate_model_claim(manifest, capture, selected, None)
        assert result.status is GateStatus.EVIDENCE_NOT_PROJECTED
        assert result.admitted == UNKNOWN_TRIPLET
        assert not result.all_three_correct
        with pytest.raises(ValueError, match="unavailable source"):
            gate_model_claim(manifest, capture, selected, triplet("A", "A", "MATCH"))


def test_oversized_real_response_never_clipped_silently() -> None:
    temp, manifest, capture = fixture(
        Case("huge-response", "B", "forced", 1),
        original_response="長" * 4500,
    )
    with temp:
        chosen = select_evidence(manifest, capture, budget_bytes=8192)
        assert chosen.status is SelectionStatus.OVER_BUDGET
        assert chosen.packet is None
        assert chosen.required_bytes > 8192


def test_complete_case_correct_model_claim_needs_external_label() -> None:
    temp, manifest, capture = fixture(Case("complete-case", "B", "forced", 1))
    with temp:
        source = select_evidence(manifest, capture)
        response = triplet("A", "B", "MISMATCH")
        checked = gate_model_claim(
            manifest, capture, source, response,
            blind_e_choice=Choice.A, independent_e_annotation=True,
        )
        assert checked.status is GateStatus.TEST_ONLY
        assert checked.raw == ModelTriplet(Choice.A, Choice.B, Relation.MISMATCH)
        assert checked.admitted == checked.raw
        assert checked.all_three_correct


def test_no_annotation_cannot_assert_original_choice_or_relation() -> None:
    temp, manifest, capture = fixture(Case("no-human", "B", "forced", 1))
    with temp:
        source = select_evidence(manifest, capture)
        wrong = gate_model_claim(
            manifest, capture, source, triplet("A", "B", "MISMATCH"),
            blind_e_choice=Choice.A, independent_e_annotation=False,
        )
        assert wrong.status is GateStatus.UNVERIFIED_E_ANNOTATION
        assert wrong.admitted == ModelTriplet(Choice.UNKNOWN, Choice.B, Relation.UNKNOWN)
        assert "unsupported_original_choice" in wrong.rejected
        assert not wrong.all_three_correct
        abstain = gate_model_claim(manifest, capture, source, triplet("UNKNOWN", "B", "UNKNOWN"))
        assert abstain.admitted == ModelTriplet(Choice.UNKNOWN, Choice.B, Relation.UNKNOWN)
        assert not abstain.all_three_correct


def test_missing_reveal_must_force_unknown_even_if_model_hallucinates_target() -> None:
    temp, manifest, capture = fixture(Case("no-e2", "B", "forced", 2, "no_reveal"))
    with temp:
        selected = select_evidence(manifest, capture)
        assert selected.packet is not None and not selected.packet.has_reveal
        assert [x["step"] for x in selected.packet.source_value()["sources"]] == ["setup", "E"]
        assert selected.packet.source_value()["reveal_state"] == "ABSENT"
        assert "事前に指定されていた参照先" not in model_question(selected)
        wrong = gate_model_claim(
            manifest, capture, selected, triplet("A", "B", "MATCH"),
            blind_e_choice=Choice.A, independent_e_annotation=True,
        )
        assert wrong.admitted == ModelTriplet(Choice.A, Choice.UNKNOWN, Relation.UNKNOWN)
        assert not wrong.all_three_correct
        assert "unsupported_revealed_target" in wrong.rejected
        assert "unsupported_relation" in wrong.rejected
        correct = gate_model_claim(
            manifest, capture, selected, triplet("A", "UNKNOWN", "UNKNOWN"),
            blind_e_choice=Choice.A, independent_e_annotation=True,
        )
        assert correct.admitted == ModelTriplet(Choice.A, Choice.UNKNOWN, Relation.UNKNOWN)
        assert correct.all_three_correct


def test_relation_fails_when_target_claim_is_wrong() -> None:
    temp, manifest, capture = fixture(Case("wrong-target", "B", "forced", 1))
    with temp:
        selected = select_evidence(manifest, capture)
        wrong = gate_model_claim(
            manifest, capture, selected, triplet("A", "A", "MATCH"),
            blind_e_choice=Choice.A, independent_e_annotation=True,
        )
        assert wrong.admitted == ModelTriplet(Choice.A, Choice.UNKNOWN, Relation.UNKNOWN)
        assert not wrong.all_three_correct


def test_relation_fails_when_original_choice_is_wrong() -> None:
    temp, manifest, capture = fixture(Case("wrong-prior", "B", "forced", 1))
    with temp:
        selected = select_evidence(manifest, capture)
        wrong = gate_model_claim(
            manifest, capture, selected, triplet("B", "B", "MATCH"),
            blind_e_choice=Choice.A, independent_e_annotation=True,
        )
        assert wrong.admitted == ModelTriplet(Choice.UNKNOWN, Choice.B, Relation.UNKNOWN)


def test_clarification_has_no_original_mismatch() -> None:
    temp, manifest, capture = fixture(Case("clarified", "B", "natural", 1))
    with temp:
        selected = select_evidence(manifest, capture)
        correct = gate_model_claim(
            manifest, capture, selected, triplet("CLARIFY", "B", "NO_CHOICE"),
            blind_e_choice=Choice.CLARIFY, independent_e_annotation=True,
        )
        assert correct.all_three_correct
        wrong = gate_model_claim(
            manifest, capture, selected, triplet("CLARIFY", "B", "MISMATCH"),
            blind_e_choice=Choice.CLARIFY, independent_e_annotation=True,
        )
        assert wrong.admitted.relation is Relation.UNKNOWN


def test_unclassifiable_original_remains_unknown() -> None:
    temp, manifest, capture = fixture(Case("unclassifiable", "B", "natural", 1))
    with temp:
        selected = select_evidence(manifest, capture)
        checked = gate_model_claim(
            manifest, capture, selected, triplet("UNKNOWN", "B", "UNKNOWN"),
            blind_e_choice=Choice.OTHER, independent_e_annotation=True,
        )
        assert checked.admitted.original_choice is Choice.UNKNOWN
        assert checked.admitted.relation is Relation.UNKNOWN
        assert checked.all_three_correct


@pytest.mark.parametrize(
    "claim",
    (
        "original_choice: A\nrevealed_target: B\nrelation: MATCH\nextra text",
        "original_choice: A\nrevealed_target: C\nrelation: MATCH",
        "original_choice: BOTH\nrevealed_target: B\nrelation: MISMATCH",
        "revealed_target: B\noriginal_choice: A\nrelation: MISMATCH",
        "original_choice: A\nrevealed_target: B\nrelation: MATCH\n",
        "original_choice=A\nrevealed_target=B\nrelation=MATCH",
    ),
)
def test_incorrect_schema_is_not_fluid_language_admission(claim: str) -> None:
    # Final newline is permissible, other extra lines / invalid enums are not.
    if claim.endswith("MATCH\n") and claim.count("\n") == 3:
        assert parse_triplet(claim) == ModelTriplet(Choice.A, Choice.B, Relation.MATCH)
    else:
        with pytest.raises(ValueError):
            parse_triplet(claim)


def test_bad_model_claim_returns_unknown_fields() -> None:
    temp, manifest, capture = fixture(Case("bad-text", "B", "natural", 0))
    with temp:
        selected = select_evidence(manifest, capture)
        invalid = gate_model_claim(
            manifest, capture, selected,
            "(1) 私は経験からBだと思ったと思います",
            blind_e_choice=Choice.A, independent_e_annotation=True,
        )
        assert invalid.status is GateStatus.MALFORMED_MODEL_CLAIM
        assert invalid.admitted == UNKNOWN_TRIPLET
        assert not invalid.all_three_correct


def test_capture_tamper_wrong_manifest_or_replay_rejected() -> None:
    temp, manifest, capture = fixture(Case("wrong-capture", "B", "natural", 0))
    with temp:
        selected = select_evidence(manifest, capture)
        changed = json.loads(json.dumps(capture))
        changed["turns"][2]["request"] = "参照先はA"
        with pytest.raises(ValueError, match="does not match"):
            gate_model_claim(manifest, changed, selected, None)
        changed = json.loads(json.dumps(capture))
        changed["case_id"] = "cross-case"
        with pytest.raises(ValueError, match="cross-case"):
            gate_model_claim(manifest, changed, selected, None)
        changed_manifest = json.loads(json.dumps(manifest))
        changed_manifest["case"]["target"] = "A"
        with pytest.raises(ValueError, match="digest mismatch"):
            select_evidence(changed_manifest, capture)
        with pytest.raises(ValueError, match="forged"):
            gate_model_claim(
                manifest, capture,
                replace(selected, packet=replace(selected.packet, case_id="forged")),
                None,
            )


def test_evaluator_packet_is_not_a_claim_of_verified_r15_authority() -> None:
    temp, manifest, capture = fixture(Case("authority", "B", "forced", 1))
    with temp:
        packet = select_evidence(manifest, capture).packet
        assert packet is not None
        assert "R15" not in packet.source_value()["authority"]
        assert all(
            "source_revision" not in x and "trusted_producer" not in x
            for x in packet.source_value()["sources"]
        )


def test_budget_uses_utf8_bytes_not_character_count() -> None:
    temp, manifest, capture = fixture(Case("utf8", "B", "forced", 1))
    with temp:
        selection = select_evidence(manifest, capture)
        assert selection.packet is not None
        encoded = json.dumps(
            selection.packet.source_value(), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        assert len(encoded) == selection.required_bytes


def test_valid_no_model_result_does_not_forge_success() -> None:
    temp, manifest, capture = fixture(Case("no-response", "A", "forced", 1))
    with temp:
        selection = select_evidence(manifest, capture)
        result = gate_model_claim(
            manifest, capture, selection, None,
            blind_e_choice=Choice.A, independent_e_annotation=True,
        )
        assert result.status is GateStatus.MALFORMED_MODEL_CLAIM
        assert result.admitted == UNKNOWN_TRIPLET
