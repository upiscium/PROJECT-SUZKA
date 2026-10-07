"""U5 source authority, privacy, and non-invasion guards."""

import ast
from dataclasses import fields, replace
from pathlib import Path

import pytest

from suzka.attention.common import ATTENTION_PROMPT_BUDGET_BYTES
from suzka.memory import DualMemorySystem
from suzka.persona import AttentionPromptPayload
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.chat_context import resolve_chat_context
from suzka.runtime.main_loop import SuzkaMainLoop, _CommittedReadViews
from test_attention_main_loop import _RecordingProvider, _runtime, _settings


ROOT = Path(__file__).resolve().parents[1]


def test_u5_document_limits_are_selected_contribution_not_whole_prompt() -> None:
    document = (ROOT / "docs" / "r14-u5-same-turn-attention.md").read_text(encoding="utf-8")
    assert str(ATTENTION_PROMPT_BUDGET_BYTES) in document
    assert "not the entire model\nprompt" in document
    assert "Human U5 focused review is pending" in document
    assert "U6, Ready, and Merge remain unauthorized" in document
    assert "U5 does **not** wire the optional U3 assessment" in document
    assert not any(
        token in item.name
        for item in fields(_CommittedReadViews)
        for token in ("payload", "text", "assessment", "metacognition")
    )


def test_u5_refresh_source_capture_has_no_raw_authority_parameters() -> None:
    tree = ast.parse((ROOT / "suzka" / "runtime" / "main_loop.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    refresh = functions["_refresh_attention_prompt"]
    assert [argument.arg for argument in refresh.args.args] == [
        "self", "event", "current_context", "emotion_state",
        "working_memory_view", "working_memory",
    ]
    calls = {
        node.func.attr
        for node in ast.walk(refresh)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "current_event" in calls and "snapshot" in calls and "refresh" in calls
    assert not {"observe_metacognition", "assess_metacognition", "generate", "retrieve_context"} & calls
    # U4's private lock/bundle bridge must not spread into MainLoop.
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"_lock", "_bundle"}
        for node in ast.walk(tree)
    )


def test_u5_producer_rejects_caller_forged_context_even_on_actual_worker(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    loop = SuzkaMainLoop(settings, _RecordingProvider(), DualMemorySystem(settings))
    runtime = _runtime()
    loop.bind_runtime(runtime)
    before = loop.agent_state_ports.attention_state_port.export_attention_state()

    def handler() -> None:
        event = runtime.current_event()
        assert event is not None
        current = resolve_chat_context(loop.context_registry, None)
        forged = replace(current, source_session_id="caller-forged-session")
        view = loop.working_memory.select_contextual(
            loop.working_memory_resolver, loop.context_registry, current.context_id
        )
        with pytest.raises(RuntimeError, match="current R09 authority"):
            loop._refresh_attention_prompt(
                event, forged, loop.emotion_engine.state, view, loop.working_memory
            )
        assert loop.agent_state_ports.attention_state_port.export_attention_state() == before

    runtime.start()
    try:
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, handler).result(timeout=10)
    finally:
        runtime.shutdown()


def test_u5_payload_is_ephemeral_sealed_and_has_distinct_byte_witnesses() -> None:
    assert {item.name for item in fields(AttentionPromptPayload)} == {
        "event", "selection_digest", "included_candidate_ids", "witnessed_bytes",
        "rendered_bytes", "rendered_text", "rendered_digest",
    }
    with pytest.raises(TypeError):
        AttentionPromptPayload()
