"""U1 dependency/absence guards and AST-checked Human obligation index."""

import ast
from pathlib import Path
import re

from suzka.r15.bounds import derive_r15_schema_budget


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "r15-u1-contract.md"
R15 = ROOT / "suzka" / "r15"


def test_r15_u1_pure_modules_are_not_runtime_producers() -> None:
    forbidden = (
        "suzka.runtime", "suzka.memory", "suzka.api", "suzka.models",
        "suzka.cognition", "suzka.decision", "suzka.action", "suzka.scheduler",
        "k" + "agya", "fastapi", "torch", "transformers", "requests",
    )
    for path in (R15 / "__init__.py", R15 / "common.py", R15 / "contracts.py", R15 / "stance.py", R15 / "bounds.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import) else [node.module or ""]
                ):
                    assert not any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden), (path, name)
            if isinstance(node, ast.ClassDef):
                assert node.name not in {
                    "RelationshipSystem", "NarrativeSelfSystem", "SelfModelSystem",
                    "AgentStateSnapshotV10", "StatePort", "Decision", "GoalSystem",
                }
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in {
                    "commit", "restore", "reconcile", "publish", "update", "mutate",
                    "refresh", "tick", "schedule", "build_prompt", "assess_metacognition",
                }
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (node.func.attr == "now" and isinstance(node.func.value, ast.Name) and node.func.value.id == "datetime")
            if isinstance(node, ast.Name):
                assert node.id not in {"uuid4", "ModelInference", "AgentRuntime", "StateWAL"}
    for path in (ROOT / "suzka").rglob("*.py"):
        if R15 in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("suzka.r15") for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("suzka.r15"), path
                if node.module == "suzka":
                    assert all(alias.name != "r15" for alias in node.names), path
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
                assert not any(isinstance(arg, ast.Constant) and arg.value == "suzka.r15" for arg in node.args), path
    main_loop = (ROOT / "suzka" / "runtime" / "main_loop.py").read_text(encoding="utf-8")
    assert "suzka.r15" not in main_loop
    assert "observe_metacognition(" not in main_loop
    assert "assess_metacognition(" not in main_loop


def test_r15_u1_obligations_index_is_complete_and_ast_checked() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    a_rows = re.findall(r"^\| (A\d+) \| ([A-Z0-9_]+) \| (.*?) \|$", document, re.MULTILINE)
    f_rows = re.findall(r"^\| (F\d+) \| ([A-Z0-9_]+) \| (.*?) \|$", document, re.MULTILINE)
    assert [name for name, _, _ in a_rows] == [f"A{number}" for number in range(1, 28)]
    assert [name for name, _, _ in f_rows] == [f"F{number}" for number in range(1, 23)]
    for name, classification, evidence in a_rows + f_rows:
        assert classification in {"U1_DIRECT", "DEFER_U2", "DEFER_U3", "DEFER_U5", "DEFER_U6",
                                  "DEFER_U2_U4", "DEFER_U2_U5", "DEFER_U2_U6", "DEFER_U5_U6",
                                  "PARTIAL_DEFER_U2", "PARTIAL_DEFER_U3", "PARTIAL_DEFER_U4",
                                  "PARTIAL_DEFER_U5", "PARTIAL_DEFER_U6", "PARTIAL_DEFER_U7",
                                  "PARTIAL_DEFER_U2_U4", "PARTIAL_DEFER_U2_U6"}, name
        if classification == "U1_DIRECT":
            assert "tests/test_r15_u1_" in evidence, name
        if classification.startswith("DEFER_"):
            assert "tests/" not in evidence, name
        for filename, function in re.findall(r"(tests/test_r15_u1_[a-z_]+\.py)::(test_[a-z0-9_]+)", evidence):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            assert function in {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}, (name, filename, function)
    budget = derive_r15_schema_budget()
    for expected in (
        f"**v10 before reserve** | **{budget.v10_before_reserve_bytes}**",
        f"**v10 including reserve** | **{budget.v10_with_reserve_bytes}**",
        f"**Margin beyond full reserve** | **{budget.remaining_beyond_reserve_bytes}**",
        "R09", "R10", "R11", "R12", "R13", "R14", "R17/R18",
        "no present production consumer", "REQUIRES_TRUSTED_ROOT",
        "current_semantic_producer", "negative zero",
    ):
        assert expected.lower() in document.lower(), expected
