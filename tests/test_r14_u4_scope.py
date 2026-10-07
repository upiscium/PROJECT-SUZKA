"""U4 schema/capacity/API guards without production Attention authority."""

import ast
from pathlib import Path

import suzka.runtime as runtime
from suzka.runtime.agent_state import (
    AGENT_STATE_FUTURE_STATE_RESERVE_BYTES,
    AGENT_STATE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V8_BASE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V8_SCHEMA_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V9_SCHEMA_FIELD_MAX_BYTES,
    AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES,
    AgentStatePorts,
    AgentStateSnapshotV8,
    AgentStateSnapshotV9,
    AttentionAgentStatePorts,
    project_agent_state_schema_max_bytes,
)
from suzka.runtime.attention_state_codec import AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES
from suzka.runtime.attention_state_port import AttentionStatePort


ROOT = Path(__file__).resolve().parents[1]


def test_u4_document_matches_exact_additive_schema_and_capacity() -> None:
    document = (ROOT / "docs" / "r14-u4-attention-persistence.md").read_text(encoding="utf-8")
    assert set(AgentStateSnapshotV9.model_fields) == set(AgentStateSnapshotV8.model_fields) | {"attention_state"}
    assert AgentStateSnapshotV9.model_fields["attention_state"].is_required()
    assert AGENT_STATE_V9_SCHEMA_FIELD_MAX_BYTES["attention_state"] == AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES == 4_193_496
    assert AGENT_STATE_V8_BASE_MAX_SERIALIZED_BYTES == 109_573_688
    assert AGENT_STATE_V8_SCHEMA_MAX_SERIALIZED_BYTES == 126_350_904
    assert AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES == 113_767_203
    assert AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES == 130_544_419
    assert AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES - AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES == AGENT_STATE_FUTURE_STATE_RESERVE_BYTES == 16_777_216
    assert AGENT_STATE_MAX_SERIALIZED_BYTES - AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES == 3_673_309
    assert project_agent_state_schema_max_bytes() == AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES
    assert project_agent_state_schema_max_bytes(schema_version=8, base_schema_version=8) == AGENT_STATE_V8_SCHEMA_MAX_SERIALIZED_BYTES
    for count in (AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES, AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES, AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES):
        assert f"{count:,}" in document
    assert "Human U4 focused review is pending" in document


def test_u4_keeps_r13_only_port_protocol_and_exports_separate_capability() -> None:
    assert set(AgentStatePorts.__annotations__) == {
        "motivation_state_port", "goal_state_port", "commitment_state_port",
    }
    assert set(AttentionAgentStatePorts.__annotations__) == {"attention_state_port"}
    assert runtime.AgentStateSnapshot is AgentStateSnapshotV9
    assert runtime.CURRENT_AGENT_STATE_SCHEMA_VERSION == 9
    assert runtime.AgentStateSnapshotV8 is AgentStateSnapshotV8
    assert runtime.AttentionStatePort is AttentionStatePort
    assert runtime.AttentionAgentStatePorts is AttentionAgentStatePorts


def test_u4_new_seams_never_call_competition_or_later_authorities() -> None:
    forbidden_imports = (
        "suzka.metacognition", "suzka.cognition", "suzka.models", "suzka.memory",
        "suzka.decision", "suzka.action", "suzka.self_model", "suzka.scheduler",
    )
    forbidden_calls = {
        "refresh", "compete_attention", "select_attention_prompt",
        "assess_metacognition", "observe_metacognition", "generate", "schedule",
    }
    for name in ("attention_state_codec.py", "attention_state_port.py"):
        path = ROOT / "suzka" / "runtime" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            else:
                names = ()
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in names for prefix in forbidden_imports
            ), (path, names)
            if isinstance(node, ast.Call):
                called = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute)
                    else None
                )
                assert called not in forbidden_calls, (path, called)


def test_u4_snapshot_has_no_assessment_or_rendered_source_fields() -> None:
    assert not any(
        token in field
        for field in AgentStateSnapshotV9.model_fields
        for token in ("metacognition", "assessment", "prompt", "competition", "rendered")
    )
