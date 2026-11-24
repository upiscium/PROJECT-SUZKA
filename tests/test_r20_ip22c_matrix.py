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


def test_the_new_tests_are_registered_and_the_f15_row_is_bound_to_the_slice_acceptance() -> None:
    expected_new = {
        "F01": 1, "F02": 2, "F05": 3, "F06": 2, "F12": 1, "F13": 1, "F14": 1,
    }
    for case in CASES:
        assert len(case.new) == expected_new.get(case.case, 0), case.case
        for ref in case.new:
            assert ref.file.startswith(("test_r20_ip22c_", "test_r20_ip22d_")), ref.node_id
    assert [case.case for case in CASES if case.smoke] == ["F06", "F15"]
    # F15: no longer PENDING; its smoke tests are the three rewritten F15-x tests, none counted
    f15 = case_for("F15")
    assert f15.status == COVERED and f15.pending_reason is None and f15.new == ()
    assert [ref.name.split("_")[1:5] for ref in f15.smoke] == [
        ["f15", "x", "1", "a"], ["f15", "x", "2", "a"], ["f15", "x", "3", "an"],
    ]
    assert all(ref.file == "test_r20_ip22d_f15.py" for ref in f15.smoke)
    assert not set(f15.smoke) & set(f15.counted)
    assert "F15" in covered_cases()
    # the counted F15 evidence is the deterministic T1 DC tests (no process tier)
    assert f15.counted and all(ref.file.startswith("test_dc") for ref in f15.counted)
    # the IP-22c files carry no F15 test: F15-x lives in IP-22d
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


def test_the_runner_writes_one_label_per_f_and_the_known_violations_with_their_text(tmp_path: Path) -> None:
    runner, calls = _fake({})
    manifest, labels = run_matrix(runner, tmp_path)
    assert set(labels) == set(EXPECTED_CASES)
    assert labels["F14"] == "PASS_TRANSITIONAL"
    assert all(labels[name] == "PASS" for name in EXPECTED_CASES if name != "F14")  # F15 included
    ran = {node_id for batch in calls for node_id in batch}
    f15 = case_for("F15")
    assert {ref.node_id for ref in f15.counted} <= ran  # F15 is RUN, not just listed
    assert {ref.node_id for ref in f15.boundary_ids} <= ran  # and so is every boundary test
    assert not {ref.node_id for ref in f15.smoke} & ran  # the T2 smoke only on request
    assert not any(ref.node_id in ran for ref in case_for("F06").smoke)
    path = manifest.write()
    written = json.loads(path.read_text(encoding="utf-8"))
    results = written["results"]
    assert results["F15"] == "PASS" and results["table"] == "PASS"
    assert results["F14.second_interpretation"] == "TRANSITIONAL"
    ids = [violation.id for violation in f15.known_violations]
    assert len(ids) >= 13 and "KV-SEMANTIC-QUARANTINE-OFF" in ids and "KV-MODE-REFUSE" in ids
    for violation_id in ids:
        assert results[f"F15.known_violation.{violation_id}"] == "NOT_COVERED_KNOWN_VIOLATION", violation_id
    # each one is also in the timeline with its D1 text and its source
    entries = [e for e in written["timeline"] if e.get("kind") == "known_violation"]
    assert [e["id"] for e in entries] == ids
    assert all(len(e["text"].split()) >= 8 and e["source"] for e in entries)
    assert all(results[f"F15.covered.{c.name}"] == "PASS" for c in f15.covered_classes)
    assert written["fixture_only"] is True and written["tier"] == "T1"
    assert f_matrix.exit_code(labels, strict=False, results=results) == 0


def test_strict_passes_only_when_every_f_is_done_and_every_uncovered_class_is_a_named_known_violation() -> None:
    done = {name: "PASS" for name in EXPECTED_CASES}
    kv = {"F15.known_violation.KV-X": "NOT_COVERED_KNOWN_VIOLATION", "F15.covered.x": "PASS"}
    assert f_matrix.exit_code(done, strict=True, results=kv) == 0
    # a PENDING row, a transitional row, an unnamed / unproven violation or a weaker covered class is not DONE
    assert f_matrix.exit_code({**done, "F15": "PENDING"}, strict=True, results=kv) == 3
    assert f_matrix.exit_code({**done, "F14": "PASS_TRANSITIONAL"}, strict=True, results=kv) == 3
    assert f_matrix.exit_code(done, strict=True, results={**kv, "F15.known_violation.KV-Y": "PASS"}) == 3
    assert f_matrix.exit_code(done, strict=True, results={**kv, "F15.covered.y": "SMOKE_NOT_COUNTED"}) == 3
    assert f_matrix.exit_code({**done, "F14": "PASS_TRANSITIONAL"}, strict=False, results=kv) == 0
    # a failed or missing result is a failure in both modes
    for strict in (False, True):
        assert f_matrix.exit_code({**done, "F15": "FAIL"}, strict=strict, results=kv) == 1
        assert f_matrix.exit_code(done, strict=strict, results={**kv, "F15.known_violation.KV-X": "FAIL"}) == 1
        assert f_matrix.exit_code(done, strict=strict, results={**kv, "F15.known_violation.KV-X": "NOT_RUN"}) == 1


def test_a_known_violation_whose_boundary_test_fails_is_not_reported_as_a_known_violation(tmp_path: Path) -> None:
    f15 = case_for("F15")
    victim = next(v for v in f15.known_violations if v.evidence)
    runner, _ = _fake({victim.evidence[0].node_id: "failed"})
    manifest, labels = run_matrix(runner, tmp_path, only=["F15"])
    results = manifest.to_dict()["results"]
    assert results[f"F15.known_violation.{victim.id}"] == "FAIL"  # the boundary moved: not "known", a failure
    assert f_matrix.exit_code(labels, strict=False, results=results) == 1
    silent, _ = _fake({})
    quiet_runner = lambda ids: {k: v for k, v in silent(ids).items() if k != victim.evidence[0].node_id}  # noqa: E731
    manifest, labels = run_matrix(quiet_runner, tmp_path, only=["F15"])
    results = manifest.to_dict()["results"]
    assert results[f"F15.known_violation.{victim.id}"] == "NOT_RUN"
    assert f_matrix.exit_code(labels, strict=False, results=results) == 1


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
    if row.known_violations:
        assert row.status == COVERED and row.acceptance and row.covered_classes


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


# ------------------------------------------------------- the F15 binding (IP-22d)

_NOT_COVERED_PHRASES = {
    "KV-JOURNAL": "Journal damage (including a torn tail",
    "KV-SUBJECT-HISTORY": "SubjectHistory loss / corruption / disablement / capacity",
    "KV-AMBIGUOUS": "AMBIGUOUS commit evidence",
    "KV-EPISODIC-AND-HISTORY-PARTICIPANTS": "memory.episodic and subject.history failures",
    "KV-UNAVAILABLE": "merely Unavailable (not divergent)",
    "KV-DB2-AND-RECEIPT-RETENTION": "DB2 projection and semantic_receipt_retention failures",
    "KV-TRUE-ROLLBACK-GATE": "true-rollback gate refusals",
    "KV-NO-OPEN-TRANSACTION": "a memory store damaged with no open transaction",
    "KV-STALE-RESTORE": "a stale restore over a quarantined store",
    "KV-HARD-STUCK": "HARD where compaction cannot free enough",
    "KV-THREADS-NOT-STOPPED": "threads that do not stop",
    "KV-PROCESS-DIES": "a process that dies",
}


def test_every_class_of_the_d1_not_covered_list_is_a_known_violation_with_its_text() -> None:
    f15 = case_for("F15")
    assert f15.acceptance is not None
    # the acceptance carries the D1 wording: bounds, the four clauses and the stated rule
    for phrase in (
        "T_failed_to_serving_max 400 s", "T_boot_max 240 s", "T_hard (600 s) + at most 5 rungs of 60 s",
        "(1) each damaged artifact is byte-preserved", "(2) no row or record copies content",
        "(3) no Action is admitted", "(4) every stop in the run ends in a terminal or re-evaluable state",
        "a known violation of 'a safety stop must never stop the subject', reported as such",
    ):
        assert phrase in f15.acceptance, phrase
    by_id = {violation.id: violation for violation in f15.known_violations}
    for violation_id, phrase in _NOT_COVERED_PHRASES.items():
        assert phrase in f15.acceptance, phrase  # the D1 'Not covered' list names it
        assert violation_id in by_id, violation_id
        assert "F15 acceptance 'Not covered'" in by_id[violation_id].source
        assert "F15 slice D1 r3 FINAL" in by_id[violation_id].source
    # the other three are the setting / implementation boundaries, each with a cited D1 place
    extra = set(by_id) - set(_NOT_COVERED_PHRASES)
    assert extra == {"KV-SEMANTIC-QUARANTINE-OFF", "KV-HARD-SHRINK-OFF", "KV-MODE-REFUSE"}
    assert all("F15 slice D1 r3 FINAL" in by_id[name].source for name in extra)
    # a class with no deterministic boundary test says so by an empty evidence list (listed anyway)
    assert [v.id for v in f15.known_violations if not v.evidence] == [
        "KV-DB2-AND-RECEIPT-RETENTION", "KV-NO-OPEN-TRANSACTION", "KV-PROCESS-DIES",
    ]


def test_semantic_quarantine_is_off_by_default_and_the_default_mode_is_refuse() -> None:
    """The claims the F15 row makes about settings, checked against the product."""

    from suzka.config.schema import DegradedContinuationSettings
    from suzka.runtime.domain_quarantine import DEFAULT_QUARANTINE_PARTICIPANTS

    assert "memory.experience" in DEFAULT_QUARANTINE_PARTICIPANTS
    assert "memory.semantic" not in DEFAULT_QUARANTINE_PARTICIPANTS  # Semantic quarantine is OFF by default
    defaults = DegradedContinuationSettings()
    assert defaults.mode == "refuse"  # F15 is NOT in force under the default config
    assert defaults.hard_band_auto_shrink is True
    caveat = case_for("F15").caveat or ""
    assert "mode = continue" in caveat and "OFF by default" in caveat and "KV-MODE-REFUSE" in caveat
    ids = {v.id for v in case_for("F15").known_violations}
    assert {"KV-SEMANTIC-QUARANTINE-OFF", "KV-MODE-REFUSE"} <= ids
    covered_names = " ".join(c.name for c in case_for("F15").covered_classes).lower()
    assert "semantic" not in covered_names  # Semantic is not claimed as covered


def _source(file: str, name: str) -> str:
    tree = ast.parse((ROOT / "tests" / file).read_text(encoding="utf-8"))
    (node,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    return ast.get_source_segment((ROOT / "tests" / file).read_text(encoding="utf-8"), node) or ""


def test_the_hard_ladder_is_exercised_only_with_hard_band_auto_shrink() -> None:
    f15 = case_for("F15")
    (hard,) = [c for c in f15.covered_classes if c.name.startswith("Journal over its hard cap")]
    assert "hard_band_auto_shrink = true" in hard.setting and "never with it off" in hard.setting
    for ref in hard.evidence:
        text = _source(ref.file, ref.name)
        assert "hard_band_auto_shrink" not in text or "hard_band_auto_shrink=False" not in text.replace(" ", ""), ref.node_id
        assert '"hard_band_auto_shrink": False' not in text, ref.node_id
    # the same row owns the opposite: with the ladder off the band blocks as before (a known violation)
    (off,) = [v for v in f15.known_violations if v.id == "KV-HARD-SHRINK-OFF"]
    assert off.evidence
    assert "hard_band_auto_shrink" in _source(off.evidence[0].file, off.evidence[0].name)


def test_the_f15_binding_checks_catch_a_missing_text_a_t2_boundary_test_and_a_vacuous_class() -> None:
    f15 = case_for("F15")
    original = f_matrix.CASES

    def with_(**changes: object) -> list[str]:
        f_matrix.CASES = tuple(dataclasses.replace(f15, **changes) if c.case == "F15" else c for c in original)  # type: ignore[arg-type]
        try:
            return check_table()
        finally:
            f_matrix.CASES = original

    textless = dataclasses.replace(f15.known_violations[0], text="too short")
    assert any("no D1 text" in p for p in with_(known_violations=(textless, *f15.known_violations[1:])))
    t2 = dataclasses.replace(
        f15.known_violations[0], evidence=(f_matrix._t("test_r20_ip22d_f15.py", f15.smoke[0].name),)
    )
    assert any("a T2 test" in p for p in with_(known_violations=(t2, *f15.known_violations[1:])))
    ghost = dataclasses.replace(f15.known_violations[0], evidence=(f_matrix._t("test_dc5_supervisor.py", "test_no_such"),))
    assert any("does not exist" in p for p in with_(known_violations=(ghost, *f15.known_violations[1:])))
    twice = (f15.known_violations[0], *f15.known_violations)
    assert any("used twice" in p for p in with_(known_violations=twice))
    assert any("without a known-violation list" in p for p in with_(known_violations=()))
    assert any("without covered classes" in p for p in with_(covered_classes=()))
    weak = dataclasses.replace(f15.covered_classes[0], evidence=(f_matrix._t("test_dc6_hard_band.py", "test_run_at_idle_runs_when_idle_and_returns_true"),))
    assert any("not a counted test" in p for p in with_(covered_classes=(weak, *f15.covered_classes[1:])))
    assert any("no acceptance wording" in p for p in with_(acceptance=None))
    assert check_table() == []
