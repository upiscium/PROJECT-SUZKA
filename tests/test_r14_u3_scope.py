"""Executable U3 API, documentation, and non-integration scope boundaries."""

import ast
from dataclasses import fields
from pathlib import Path

import suzka.metacognition as metacognition
from suzka.attention.bounds import derive_attention_schema_size_budget
from suzka.attention.contracts import AttentionContinuity
from suzka.metacognition.assessment import METACOGNITION_QUALITY_WEIGHTS
from suzka.metacognition.evidence import (
    METACOGNITION_MAX_OBSERVATION_BYTES,
    MetacognitionObservation,
    derive_metacognition_observation_max_bytes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_u3_documented_policy_and_ephemeral_envelope_match_derivation() -> None:
    document = (ROOT / "docs" / "r14-u3-metacognition.md").read_text(encoding="utf-8")
    assert f"**{METACOGNITION_MAX_OBSERVATION_BYTES} bytes**" in document
    assert METACOGNITION_MAX_OBSERVATION_BYTES == derive_metacognition_observation_max_bytes()
    assert str(METACOGNITION_QUALITY_WEIGHTS) in document
    budget = derive_attention_schema_size_budget()
    assert f"**{budget.candidate_count}**" in document
    assert f"**{budget.attention_state_max_bytes} bytes**" in document
    assert "never\nemits `SUFFICIENT`" in document
    assert "Human U3 review is pending" in document


def test_u3_facade_exports_only_the_pure_producer_and_assessor() -> None:
    assert metacognition.observe_metacognition.__module__ == "suzka.metacognition.evidence"
    assert metacognition.assess_metacognition.__module__ == "suzka.metacognition.assessment"
    assert metacognition.MetacognitionObservation is MetacognitionObservation
    assert not hasattr(metacognition, "MetacognitionSystem")


def test_u3_modules_have_no_runtime_or_mutation_authority() -> None:
    prefixes = (
        "suzka.runtime", "suzka.memory", "suzka.cognition", "suzka.models",
        "suzka.providers", "suzka.decision", "suzka.action", "suzka.self_model",
        "suzka.scheduler", "torch", "transformers", "sqlite3", "pickle",
    )
    forbidden_calls = {
        "refresh", "restore_snapshot", "compete_attention", "select_attention_prompt",
        "tick", "schedule", "commit", "persist", "open", "write_text", "write_bytes",
    }
    for name in ("evidence.py", "assessment.py", "__init__.py"):
        path = ROOT / "suzka" / "metacognition" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports = (node.module or "",)
            else:
                imports = ()
            assert not any(
                imported == prefix or imported.startswith(prefix + ".")
                for imported in imports for prefix in prefixes
            ), (path, imports)
            if isinstance(node, ast.ClassDef):
                assert node.name == "MetacognitionObservation", (path, node.name)
            if isinstance(node, ast.Call):
                called = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute)
                    else None
                )
                assert called not in forbidden_calls, (path, called)


def test_u3_values_add_no_history_traits_or_attention_persistence() -> None:
    observation_fields = {item.name for item in fields(MetacognitionObservation)}
    for name in observation_fields:
        assert not any(token in name for token in (
            "history", "calibration", "trait", "bias", "capability", "action", "recommend",
        )), name
    assert not any(
        "metacog" in item.name or "assessment" in item.name
        for item in fields(AttentionContinuity)
    )
