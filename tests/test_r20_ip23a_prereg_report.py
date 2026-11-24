"""IP-23a: the pre-registration (create-once, immutable, hashed) and the report
(never an R20 result).  No model, no process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest
from r20_ip23_helpers import FOUR, MODEL_ID, REVISION, make_cache, make_prereg, real_model, refusal_code

from experiments.r20 import f_matrix, prereg
from experiments.r20.caps import Caps
from experiments.r20.f1_table import F1_PREREGISTRATION_TEXT
from experiments.r20.guards import GuardRefusal
from experiments.r20.report import (
    LABELS,
    STAGE_JUDGEMENT,
    build_report,
    defect_labels,
    write_report,
)


def test_the_preregistration_is_created_once_hashed_and_read_only(tmp_path: Path) -> None:
    content, digest = make_prereg(tmp_path / "base", make_cache(tmp_path / "hf"))
    base = tmp_path / "base"
    path = base / prereg.PREREG_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert (base / prereg.HASH_NAME).read_text().strip() == digest == prereg.sha256_hex(path.read_bytes())
    loaded, again = prereg.load(base)
    assert again == digest and loaded == json.loads(path.read_text())
    # created ONCE: a second record, even an identical one, is refused and the file is untouched
    before = path.read_bytes()
    assert refusal_code(prereg.create, base, content) == "PREREG_EXISTS"
    assert path.read_bytes() == before


def test_an_altered_or_writable_record_is_refused_at_every_load(tmp_path: Path) -> None:
    make_prereg(tmp_path / "base", make_cache(tmp_path / "hf"))
    base = tmp_path / "base"
    path = base / prereg.PREREG_NAME
    os.chmod(path, 0o600)
    assert refusal_code(prereg.load, base) == "PREREG_WRITABLE"
    path.write_bytes(path.read_bytes().replace(b'"attempts_max": 3', b'"attempts_max": 9'))
    os.chmod(path, 0o400)
    assert refusal_code(prereg.load, base) == "PREREG_ALTERED"
    assert refusal_code(prereg.load, tmp_path / "empty") == "PREREG_MISSING"


def test_the_record_holds_what_the_plan_says(tmp_path: Path) -> None:
    content, _ = make_prereg(tmp_path / "base", make_cache(tmp_path / "hf"))
    assert content["git_head"] == "a" * 40
    assert content["model"]["model_id"] == MODEL_ID and content["model"]["revision_pinned"] == REVISION  # type: ignore[index]
    assert content["f1"]["text"] == F1_PREREGISTRATION_TEXT  # type: ignore[index]
    assert "K = 3" in content["s0"]["k_rule"] and content["s0"]["attempts_max"] == 3  # type: ignore[index]
    assert content["s0"]["expected_structural_counts"] == {  # type: ignore[index]
        "row4_decisions": 1, "intents": 1, "execution_claims": 1, "target_effects": 1, "feedback": 1,
        "feedback_admitted_wakes": 1, "outbox_messages": 1,
    }
    caps = content["caps"]
    assert caps["max_subjects"] == 42 and caps["total_seconds"] == 36000 and caps["session_seconds"] == 9000  # type: ignore[index]
    assert caps["turn_seconds"] == 120 and caps["min_free_bytes"] == 14 * 1024**3  # type: ignore[index]
    assert content["s1"]["dc_mode"] == "refuse"  # type: ignore[index]
    assert "DC-5 mode refuse" in content["s0"]["template"]  # type: ignore[index]
    for key in ("config_template_sha256", "request_templates_sha256"):
        assert len(content[key]) == 64  # type: ignore[arg-type]
    known = content["known_violations"]
    assert [case["case"] for case in known["f_matrix"]] == list(f_matrix.EXPECTED_CASES)  # type: ignore[index]
    assert {item["label"] for item in known["f15_known_violations"]} == {"NOT_COVERED_KNOWN_VIOLATION"}  # type: ignore[index]
    assert known["four_unlisted_known_violations"] == FOUR  # type: ignore[index]
    assert "TRANSITIONAL" in known["f14_transitional"] and "partial" in known["f12_partial_coverage"]  # type: ignore[index]
    assert "no R20 Stage result" in content["not_claimed"]  # type: ignore[operator]


def test_the_humans_four_unlisted_known_violations_are_required(tmp_path: Path) -> None:
    for bad in ([], FOUR[:3], [*FOUR[:3], "TODO"], [*FOUR[:3], FOUR[0]], [*FOUR[:3], "tbd"], [*FOUR, "a fifth one of them"],
                [*FOUR[:3], "short"]):
        with pytest.raises(GuardRefusal) as refused:
            prereg.build_content(git_head="a" * 40, model=real_model(), caps=Caps(), four_violations=bad, master_seed=1,
                                 trial_id="t")
        assert refused.value.code == "FOUR_KNOWN_VIOLATIONS_REQUIRED"
        if len(bad) != 4:
            continue
        assert "V1" in refused.value.detail  # the agent's candidates are offered to choose from, never filled in
    base = tmp_path / "base"
    base.mkdir()
    content, _ = make_prereg(tmp_path / "ok", make_cache(tmp_path / "hf"))
    broken = json.loads(json.dumps(content))
    broken["known_violations"]["four_unlisted_known_violations"] = []
    assert refusal_code(prereg.create, base, broken) == "FOUR_KNOWN_VIOLATIONS_REQUIRED"
    assert not (base / prereg.PREREG_NAME).exists()
    incomplete = {key: value for key, value in content.items() if key != "caps"}
    assert refusal_code(prereg.create, base, incomplete) == "PREREG_INCOMPLETE"


def test_the_f1_plan_is_derived_before_the_first_run_and_is_deterministic() -> None:
    plan = prereg.derive_f1_plan(42, "trial-1")
    assert plan == prereg.derive_f1_plan(42, "trial-1") and plan != prereg.derive_f1_plan(43, "trial-1")
    pairs = plan["pairs"]
    assert len(pairs) == 12 and plan["whole_pair_retry_cap"] == 3
    kinds = [pair["kind"] for pair in pairs]
    assert kinds.count("TREATMENT") == 8 and kinds.count("CONTROL_NEUTRAL") == 2 and kinds.count("CONTROL_UNCERTAINTY") == 2
    assert len({pair["seed"] for pair in pairs}) == 12  # 12 distinct seeds
    firsts = [pair["first_branch"] for pair in pairs]
    assert firsts.count("NEUTRAL") == firsts.count("UNCERTAINTY") == 6  # counterbalanced
    assert sorted(plan["trial_order"]) == list(range(1, 13))
    assert refusal_code(prereg.derive_f1_plan, "x", "t") == "F1_PLAN_INPUT"


# ------------------------------------------------------------------- the report


def _results() -> list[dict[str, object]]:
    attempt = {
        "attempt": 1, "qualifies_as_s0": True, "stop_reason": None,
        "first_row4": {"row": "4", "interpretation": "SETTLED", "stage": "S2", "receipts": []},
        "step9_structural": {"equal": True, "counts": {"row4_decisions": 1}},
        "kill": {"checks": {"replay_of_every_sealed_input_equal": True}},
        "journal": {"live_bytes": 100, "soft_band_start_bytes": 1000},
    }
    bad = {**attempt, "attempt": 2, "qualifies_as_s0": False, "stop_reason": "TURN_CAP_EXCEEDED",
           "kill": {"checks": {"replay_of_every_sealed_input_equal": False}},
           "step9_structural": {"equal": False, "counts": {}, "differing": {"intents": {"expected": 1, "observed": 2}}}}
    return [
        {"stage": "G1", "g1": {"label": "GATE_PASSED", "x_not_skipped_as_neutral": True, "serving_after_s": 61.0,
                               "gpu": {"min_free": 1}, "max_turn_seconds": 20, "turn_seconds": [1, 20]}},
        {"stage": "S0", "s0": {"s0": "reached", "attempts": [attempt, bad], "negative_variant": {"label": "VARIANT_HELD"}}},
        {"stage": "F1", "f1": {"pairs": [{"pair": 1, "kind": "TREATMENT", "class": "CONFOUNDED", "members": [
            {"arm": "NEUTRAL", "sealed_row": "4", "defect": "GATE_CONFORMANCE_DEFECT"},
            {"arm": "UNCERTAINTY", "sealed_row": "4b", "infra": "OSError"}]}],
            "summary": {"claim_ceiling": "PARTIAL_VALID_EVIDENCE", "i_to_s": "DESCRIPTIVE_ONLY", "e_r15": None,
                        "valid_treatment_pairs": 0, "control_discordance": 0, "control_pairs_valid": 0, "classes": {}}}},
    ]


def _base_with_results(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    make_prereg(base, make_cache(tmp_path / "hf"))
    for index, result in enumerate(_results()):
        (base / f"results-{result['stage']}-{index}.json").write_text(json.dumps(result))
    return base


def test_the_report_shows_the_runs_the_labels_and_the_known_violations(tmp_path: Path) -> None:
    base = _base_with_results(tmp_path)
    text = build_report(base)
    assert STAGE_JUDGEMENT in text
    assert "S0: reached" in text and "GATE_PASSED" in text and "PARTIAL_VALID_EVIDENCE" in text
    for case in f_matrix.EXPECTED_CASES:
        assert case in text
    assert "NOT_COVERED_KNOWN_VIOLATION" in text and "KV-JOURNAL" in text and "Four unlisted known violations" in text
    for item in FOUR:
        assert item in text
    assert "COGNITIVE_VARIATION" in text and "HARNESS_BRIDGE" in text and "permanent core-store faults" in text
    assert "ledger: 0 subjects started" in text


def test_the_report_never_states_a_pass_of_r20(tmp_path: Path) -> None:
    import re

    text = build_report(_base_with_results(tmp_path))
    assert not re.search(r"\bPASS(ED)?\b", text) and not re.search(r"(?i)r20[^.\n]{0,24}\bpass", text)
    path = write_report(tmp_path / "base")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.read_text() == text
    # a text that would state it is refused, not written
    import experiments.r20.report as report

    original = report.build_report
    report.build_report = lambda _base: "R20 PASS\n"  # type: ignore[assignment]
    try:
        assert refusal_code(report.write_report, tmp_path / "base", tmp_path / "x.md") == "REPORT_STATES_A_PASS"
    finally:
        report.build_report = original  # type: ignore[assignment]
    assert not (tmp_path / "x.md").exists()


def test_the_labels_follow_deterministic_rules_only() -> None:
    labelled = defect_labels(_results())
    assert set(labelled) == set(LABELS)
    assert len(labelled["GATE_CONFORMANCE_DEFECT"]) == 1
    assert len(labelled["ENGINEERING_DEFECT"]) == 1  # a sealed input that does not replay
    assert labelled["COGNITIVE_VARIATION"] == []  # never assigned by the code: the Human's reading
    assert any("step 9" in item for item in labelled["UNDETERMINED"]) and any("TURN_CAP" in item for item in labelled["UNDETERMINED"])
    assert any("OSError" in item for item in labelled["UNDETERMINED"])


def test_the_report_needs_the_preregistration(tmp_path: Path) -> None:
    assert refusal_code(build_report, tmp_path) == "PREREG_MISSING"
