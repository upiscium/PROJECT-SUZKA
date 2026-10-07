"""Executable U1 documentation, dependency and non-production scope guards."""

import ast
from dataclasses import fields
from pathlib import Path

from suzka.attention.bounds import (
    ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES,
    ATTENTION_AGENT_STATE_MAX_BYTES,
    ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES,
    ATTENTION_ROOT_FIELD_OVERHEAD_BYTES,
    derive_attention_schema_size_budget,
)
from suzka.attention.common import (
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_RENDERED_ITEM_BYTES,
    ATTENTION_MAX_REVISION_HISTORY,
    ATTENTION_PROMPT_BUDGET_BYTES,
)
from suzka.attention.contracts import AttentionContinuity
from suzka.attention.policy import ATTENTION_PROMPT_FRAME_BYTES
from suzka.metacognition.contracts import (
    METACOGNITION_MAX_ASSESSMENT_BYTES,
    MetacognitiveAssessment,
)


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "r14-u1-contract.md"


def test_u1_documented_capacity_and_resources_match_executable_derivation() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    budget = derive_attention_schema_size_budget()
    assert f"universe is **{budget.candidate_count}**" in document
    assert f"Focus capacity: **{ATTENTION_MAX_FOCUS}**" in document
    assert f"**{ATTENTION_MAX_REVISION_HISTORY}**, receipts to **{ATTENTION_MAX_EVENT_RECEIPTS}**" in document
    assert f"budget: **{ATTENTION_PROMPT_BUDGET_BYTES} bytes**" in document
    assert f"framing: **{ATTENTION_PROMPT_FRAME_BYTES} bytes**" in document
    assert f"witness: **{ATTENTION_MAX_RENDERED_ITEM_BYTES} bytes**" in document
    assert f"ceiling is **{METACOGNITION_MAX_ASSESSMENT_BYTES} bytes**" in document
    assert f"| **Complete Attention JSON value** | **{budget.attention_state_max_bytes}** |" in document
    assert f"| New `attention_state` root field overhead | {ATTENTION_ROOT_FIELD_OVERHEAD_BYTES} |" in document
    before_reserve = (
        ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES
        + budget.attention_state_max_bytes
        + ATTENTION_ROOT_FIELD_OVERHEAD_BYTES
    )
    with_reserve = before_reserve + ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES
    assert f"| **v9 before reserve** | **{before_reserve}** |" in document
    assert f"| **v9 including reserve** | **{with_reserve}** |" in document
    assert f"| **Remaining margin beyond reserve** | **{ATTENTION_AGENT_STATE_MAX_BYTES - with_reserve}** |" in document


def test_u1_modules_do_not_import_runtime_providers_or_later_authorities() -> None:
    # Keep class/mutation exclusions scoped to U1 definitions. The shared
    # package façades now re-export authorized U2/U3, but must remain free of
    # production runtime/provider/later-authority dependencies.
    modules = (
        *(ROOT / "suzka" / "attention" / name for name in (
            "__init__.py", "common.py", "contracts.py", "bounds.py", "adapters.py", "policy.py"
        )),
        *(ROOT / "suzka" / "metacognition" / name for name in (
            "__init__.py", "contracts.py"
        )),
        ROOT / "suzka" / "working_memory_contracts.py",
        ROOT / "suzka" / "context_contracts.py",
        ROOT / "suzka" / "emotion_contracts.py",
    )
    forbidden_prefixes = (
        "suzka.runtime", "suzka.memory", "suzka.cognition", "suzka.models",
        "suzka.decision", "suzka.self_model", "suzka.relationship",
        "suzka.action", "suzka.scheduler", "torch", "transformers", "peft",
    )
    forbidden_classes = {"AttentionSystem", "MetacognitionSystem", "AgentStateSnapshotV9"}
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            for name in names:
                assert not any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in forbidden_prefixes
                ), (path, name)
            if isinstance(node, ast.ClassDef):
                assert node.name not in forbidden_classes, (path, node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in {"assess", "refresh", "tick", "schedule"}, (path, node.name)


def test_metacognition_is_neither_persisted_history_nor_a_decision_contract() -> None:
    continuity_fields = {field.name for field in fields(AttentionContinuity)}
    assessment_fields = {field.name for field in fields(MetacognitiveAssessment)}
    assert not any("metacog" in name or "assessment" in name for name in continuity_fields)
    assert not any(
        token in name
        for name in assessment_fields
        for token in ("history", "capability", "trait", "bias", "action", "recommend", "outcome")
    )
