"""IP-22c: the F01-F15 matrix meta-test (D1 sec. 6, "Meta-test (22c)").

The table lives in ``experiments/r20/f_matrix.py``.  These tests check the TABLE
and the RUNNER logic (with an injected fake pytest runner); the cited tests
themselves run in their own files.  Every claim of "covered" must rest on a
test that exists, is not skipped, and is T1.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from experiments.r20 import f_matrix
from experiments.r20.f_matrix import (
    CASES,
    COVERED,
    DETERMINISM,
    EXPECTED_CASES,
    PENDING,
    TRANSITIONAL,
    FCase,
    case_for,
    check_table,
    covered_cases,
    run_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_table_is_exactly_f01_to_f15_and_has_no_problem() -> None:
    assert tuple(case.case for case in CASES) == EXPECTED_CASES
    assert check_table() == []


def test_every_f_has_an_owning_unit_a_determinism_flag_and_a_test_id() -> None:
    for case in CASES:
        assert case.owning_unit.strip(), case.case
        assert case.determinism in DETERMINISM, case.case
        assert case.all_ids, case.case


def test_every_cited_test_exists_by_function_name_and_none_is_skipped() -> None:
    cited = 0
    for case in CASES:
        for ref in case.all_ids:
            path = ROOT / "tests" / ref.file
            assert path.is_file(), ref.node_id
            tree = ast.parse(path.read_text(encoding="utf-8"))
            nodes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == ref.name
            ]
            assert len(nodes) == 1, ref.node_id  # present, and unambiguous
            for decorator in nodes[0].decorator_list:
                assert "skip" not in ast.unparse(decorator) and "xfail" not in ast.unparse(decorator), ref.node_id
            cited += 1
    assert cited > 100
    # cited by FUNCTION NAME: a line number or a bare file is never an id
    for case in CASES:
        for ref in case.all_ids:
            assert ref.name.startswith("test") and ":" not in ref.name and ref.name.isidentifier()


def test_a_renamed_or_vanished_test_fails_the_matrix() -> None:
    first = case_for("F01")
    broken = dataclasses.replace(first, merged=(f_matrix._t("test_r16_u2a_source_owner.py", "test_no_such_function"),))
    patched = tuple(broken if case.case == "F01" else case for case in CASES)
    original = f_matrix.CASES
    f_matrix.CASES = patched
    try:
        assert any("F01" in problem and "does not exist" in problem for problem in check_table())
        gone = dataclasses.replace(first, merged=(f_matrix._t("test_no_such_file_at_all.py", "test_x"),))
        f_matrix.CASES = tuple(gone if case.case == "F01" else case for case in original)
        assert any("does not exist" in problem for problem in check_table())
        empty = dataclasses.replace(first, merged=(), new=(), smoke=())
        f_matrix.CASES = tuple(empty if case.case == "F01" else case for case in original)
        assert any("no test id" in problem for problem in check_table())
        flagless = dataclasses.replace(first, determinism="MAYBE")
        f_matrix.CASES = tuple(flagless if case.case == "F01" else case for case in original)
        assert any("determinism" in problem for problem in check_table())
        unowned = dataclasses.replace(first, owning_unit=" ")
        f_matrix.CASES = tuple(unowned if case.case == "F01" else case for case in original)
        assert any("no owning unit" in problem for problem in check_table())
    finally:
        f_matrix.CASES = original
    assert check_table() == []


def test_only_t1_results_count_as_covered() -> None:
    f06 = case_for("F06")
    assert f06.smoke, "F06-c is a T2 smoke"
    assert not set(f06.smoke) & set(f06.counted)
    # a T2 test moved into a counted list is a table problem
    original = f_matrix.CASES
    try:
        moved = dataclasses.replace(f06, new=f06.smoke, smoke=())
        f_matrix.CASES = tuple(moved if case.case == "F06" else case for case in original)
        assert any("uses the process tier" in problem for problem in check_table())
        # and a smoke list holding a T1 test is a problem too
        wrong = dataclasses.replace(case_for("F05"), smoke=case_for("F05").new)
        f_matrix.CASES = tuple(wrong if case.case == "F05" else case for case in original)
        assert any("smoke test" in problem and "not a T2 test" in problem for problem in check_table())
    finally:
        f_matrix.CASES = original
    # no counted test of any row uses the process tier
    for case in CASES:
        for ref in case.counted:
            tree = ast.parse((ROOT / "tests" / ref.file).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == ref.name:
                    text = ast.unparse(node)
                    assert "Tier.T2" not in text and "ChildProcess" not in text, ref.node_id
    assert "F06" in covered_cases()  # covered by its merged T1 tests, not by the smoke


def test_the_new_ip22c_tests_are_registered_and_the_f15_row_is_pending_with_none() -> None:
    expected_new = {
        "F01": 1, "F02": 2, "F05": 3, "F12": 1, "F13": 1, "F14": 1,
    }
    for case in CASES:
        assert len(case.new) == expected_new.get(case.case, 0), case.case
        for ref in case.new:
            assert ref.file.startswith("test_r20_ip22c_"), ref.node_id
    assert [case.case for case in CASES if case.smoke] == ["F06"]
    # F15: a clearly marked PENDING row; nothing is claimed and nothing new exists
    f15 = case_for("F15")
    assert f15.status == PENDING and f15.new == () and f15.smoke == ()
    assert f15.pending_reason is not None and "7103463226" in f15.pending_reason
    assert "F15" not in covered_cases()
    for path in sorted((ROOT / "tests").glob("test_r20_ip22c_*.py")):
        names = [n.name for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)]
        assert not [name for name in names if name.startswith("test_f15")], path.name


def test_f14_is_a_transitional_guard_and_not_covered_as_final_behaviour() -> None:
    f14 = case_for("F14")
    assert f14.status == TRANSITIONAL
    assert f14.transitional is not None
    assert f_matrix.TRANSITIONAL_MARK in f14.transitional and "#463" in f14.transitional
    assert "F14" not in covered_cases()
    # the test never pins the refusal as final: it states the transitional label and
    # does not assert a refusal exception type
    text = (ROOT / "tests" / "test_r20_ip22c_f14.py").read_text(encoding="utf-8")
    assert "TRANSITIONAL" in text and "pytest.raises(" not in text
    assert "NarrativeEventConflict" not in text


# ------------------------------------------------------------------ the runner


def _fake(outcomes: dict[str, str]):  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    def runner(node_ids):  # type: ignore[no-untyped-def]
        calls.append(list(node_ids))
        return {node_id: outcomes.get(node_id, "passed") for node_id in node_ids}

    return runner, calls


def test_the_runner_writes_one_label_per_f_and_never_runs_or_passes_a_pending_row(tmp_path: Path) -> None:
    runner, calls = _fake({})
    manifest, labels = run_matrix(runner, tmp_path)
    assert set(labels) == set(EXPECTED_CASES)
    assert labels["F15"] == "PENDING"
    assert labels["F14"] == "PASS_TRANSITIONAL"
    assert all(labels[name] == "PASS" for name in EXPECTED_CASES if name not in {"F14", "F15"})
    ran = {node_id for batch in calls for node_id in batch}
    elsewhere = {ref.node_id for case in CASES if case.case != "F15" for ref in case.all_ids}
    exclusive = {ref.node_id for ref in case_for("F15").all_ids} - elsewhere
    assert exclusive and not exclusive & ran  # F15's own cited tests are not run as F15 evidence
    assert not any(ref.node_id in ran for ref in case_for("F06").smoke)  # smoke only on request
    path = manifest.write()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["results"]["F15"] == "PENDING" and written["results"]["table"] == "PASS"
    assert written["results"]["F14.second_interpretation"] == "TRANSITIONAL"
    assert written["fixture_only"] is True and written["tier"] == "T1"
    assert f_matrix.exit_code(labels, strict=False) == 0
    assert f_matrix.exit_code(labels, strict=True) == 3  # pending is not a pass


def test_a_failed_skipped_or_missing_result_is_never_a_pass(tmp_path: Path) -> None:
    target = case_for("F05").new[0].node_id
    for bad in ("failed", "skipped", "error"):
        runner, _ = _fake({target: bad})
        _manifest, labels = run_matrix(runner, tmp_path, only=["F05"])
        assert labels["F05"] == "FAIL", bad
        assert f_matrix.exit_code(labels, strict=False) == 1

    def silent(_ids):  # type: ignore[no-untyped-def]
        return {}

    _manifest, labels = run_matrix(silent, tmp_path, only=["F01"])
    assert labels["F01"] == "NOT_RUN"  # an unreported test is not a pass
    assert f_matrix.exit_code(labels, strict=False) == 1


def test_a_smoke_result_is_labelled_and_never_raises_a_row(tmp_path: Path) -> None:
    smoke = case_for("F06").smoke[0].node_id
    runner, calls = _fake({smoke: "failed"})
    manifest, labels = run_matrix(runner, tmp_path, only=["F06"], include_smoke=True)
    assert any(smoke in batch for batch in calls)
    assert labels["F06"] == "PASS"  # the counted T1 tests decide, not the smoke
    assert manifest.to_dict()["results"]["F06.smoke"].startswith("SMOKE_NOT_COUNTED")
    assert manifest.to_dict()["results"]["F06.smoke"].endswith("FAIL")


def test_the_runner_command_is_the_documented_entry_point() -> None:
    assert f_matrix.main(["check"]) == 0
    source = (ROOT / "experiments" / "r20" / "f_matrix.py").read_text(encoding="utf-8")
    assert "python -m experiments.r20.f_matrix run" in source
    assert f_matrix.__name__ == "experiments.r20.f_matrix"


def test_nothing_under_suzka_imports_the_f_matrix() -> None:
    for path in (ROOT / "suzka").rglob("*.py"):
        assert "experiments.r20" not in path.read_text(encoding="utf-8"), path
    assert isinstance(CASES[0], FCase)
    assert COVERED in {case.status for case in CASES}


@pytest.mark.parametrize("case", EXPECTED_CASES)
def test_each_row_is_consistent(case: str) -> None:
    row = case_for(case)
    if row.status == COVERED:
        assert row.counted and not row.transitional and not row.pending_reason
    if row.status == PENDING:
        assert row.pending_reason and not row.new


def test_parametrized_results_aggregate_and_the_real_pytest_runner_reports_them() -> None:
    node = "tests/test_x.py::test_p"
    assert f_matrix.aggregate_outcome({f"{node}[a]": "passed", f"{node}[b]": "passed"}, node) == "passed"
    assert f_matrix.aggregate_outcome({f"{node}[a]": "passed", f"{node}[b]": "failed"}, node) == "failed"
    assert f_matrix.aggregate_outcome({f"{node}[a]": "passed", f"{node}[b]": "skipped"}, node) == "skipped"
    assert f_matrix.aggregate_outcome({"tests/test_x.py::test_pq": "passed"}, node) == f_matrix.NOT_RUN
    # the real runner on a small parametrized test of this file
    cited = "tests/test_r20_ip22c_matrix.py::test_each_row_is_consistent"
    outcomes = f_matrix.pytest_runner([cited])
    assert len([key for key in outcomes if key.startswith(cited + "[")]) == 15
    assert f_matrix.aggregate_outcome(outcomes, cited) == "passed"
    # every parametrized test the table cites is therefore not NOT_RUN: check the two known ones
    assert any(ref.name == "test_f05_w_the_stand_in_is_a_guarded_client_with_and_without_fetch_metadata" for ref in case_for("F05").new)
