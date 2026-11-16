"""#417 PR-B: AgentState V11 carries the reference-only ConversationTrace."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from suzka.api.server import create_app
from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS
from suzka.learning import AdapterRegistry
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE
from suzka.memory import DualMemorySystem
from suzka.runtime import (
    AgentStateLoadError,
    AgentStateSnapshotV10,
    AgentStateSnapshotV11,
    project_agent_state_v10_schema_max_bytes,
    project_agent_state_v11_schema_max_bytes,
)
from suzka.runtime.agent_state import (
    AGENT_STATE_FUTURE_STATE_RESERVE_BYTES,
    AGENT_STATE_MAX_SERIALIZED_BYTES,
)
from suzka.runtime.conversation_trace import (
    MAX_TRACE_CONTEXTS,
    MAX_TRACE_COUNTER,
    MAX_TRACE_ITEMS_TOTAL,
    ConversationTraceState,
    TraceTurnKind,
    conversation_trace_schema_max_bytes,
)

from test_fastapi_backend import ThinkingProvider, _client, _settings



def _settings_with(tmp_path: Path, *, enabled: bool):  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    return settings.model_copy(
        update={
            "conversation_trace": settings.conversation_trace.model_copy(
                update={"enabled": enabled}
            )
        }
    )


def _chat(client, message: str = "hello") -> dict:  # type: ignore[no-untyped-def]
    response = client.post("/api/chat", json={"message": message, "attachments": []})
    assert response.status_code == 200
    return response.json()


def _restart(settings):  # type: ignore[no-untyped-def]
    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    from fastapi.testclient import TestClient

    return TestClient(app)


def _plant_item(client) -> ConversationTraceState:  # type: ignore[no-untyped-def]
    """Put one item into the live trace, as PR-C's Chat wiring will."""

    trace = client.app.state.main_loop.conversation_trace
    store = client.app.state.agent_state_store
    sequence = store.load().last_processed_event_sequence
    trace.add(
        "conversation.default",
        episode_id="episode-planted",
        turn_kind=TraceTurnKind.CHAT,
        # Committed state may not be in the future of the next save.
        requested_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        event_sequence=sequence,
        surprise=0.5,
    )
    return trace.state


def test_disabled_by_default_keeps_v10(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _chat(client)
        snapshot = client.app.state.agent_state_store.load()
        assert type(snapshot) is AgentStateSnapshotV10
        assert client.app.state.main_loop.conversation_trace.persists is False


def test_enabled_commits_v11_and_restores_after_restart(tmp_path: Path) -> None:
    settings = _settings_with(tmp_path, enabled=True)
    with _client(tmp_path, settings=settings) as client:
        _chat(client, "first")
        planted = _plant_item(client)
        _chat(client, "second")
        snapshot = client.app.state.agent_state_store.load()
        assert type(snapshot) is AgentStateSnapshotV11
        assert ConversationTraceState.from_value(snapshot.conversation_trace) == planted
        raw = json.loads(settings.agent_state.path.read_text())
        assert raw["schema_version"] == 11
    with _restart(settings) as restarted:
        trace = restarted.app.state.main_loop.conversation_trace
        assert trace.state == planted
        assert trace.persists and not trace.frozen


def test_disabling_after_v11_freezes_and_never_downgrades(tmp_path: Path) -> None:
    settings = _settings_with(tmp_path, enabled=True)
    with _client(tmp_path, settings=settings) as client:
        _chat(client)
        planted = _plant_item(client)
        _chat(client)
    disabled = settings.model_copy(
        update={
            "conversation_trace": settings.conversation_trace.model_copy(
                update={"enabled": False}
            )
        }
    )
    with _restart(disabled) as restarted:
        trace = restarted.app.state.main_loop.conversation_trace
        assert trace.frozen and trace.persists
        _chat(restarted, "still works")
        snapshot = restarted.app.state.agent_state_store.load()
        assert type(snapshot) is AgentStateSnapshotV11
        assert ConversationTraceState.from_value(snapshot.conversation_trace) == planted


def test_v11_restore_requires_the_trace_owner(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        _chat(client)
        store = client.app.state.agent_state_store
        snapshot = store.load()
        main_loop = client.app.state.main_loop
        owner = main_loop.conversation_trace
        main_loop.conversation_trace = None
        try:
            with pytest.raises(AgentStateLoadError):
                store.restore_into(main_loop, snapshot)
        finally:
            main_loop.conversation_trace = owner


def test_restoring_an_older_schema_clears_the_v11_flag(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _chat(client)
        store = client.app.state.agent_state_store
        v10 = store.load()
        trace = client.app.state.main_loop.conversation_trace
        trace.restore_committed(ConversationTraceState())
        assert trace.persists
        # A failure restore to a committed V10 snapshot must clear the flag.
        store.restore_into(client.app.state.main_loop, v10)
        assert trace.persists is False
        assert trace.state == ConversationTraceState()


def test_a_failed_restore_rolls_the_trace_back(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        _chat(client)
        planted = _plant_item(client)
        store = client.app.state.agent_state_store
        main_loop = client.app.state.main_loop
        trace = main_loop.conversation_trace
        _chat(client)
        snapshot = store.load()
        other = snapshot.model_copy(
            update={"conversation_trace": ConversationTraceState().canonical_value()}
        )
        marker = trace.committed_marker()

        def explode(_state: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(trace, "restore_committed", explode)
        with pytest.raises(AgentStateLoadError):
            store.restore_into(main_loop, other)
        monkeypatch.undo()
        assert trace.committed_marker() == marker
        assert trace.state == planted


def test_v11_rejects_future_items_and_text_keys(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        _chat(client)
        _plant_item(client)
        _chat(client)
        snapshot = client.app.state.agent_state_store.load()
    raw = snapshot.model_dump(mode="python")
    future = json.loads(json.dumps(raw["conversation_trace"]))
    item = future["contexts"][0]["items"][0]
    late = (snapshot.saved_at + timedelta(hours=1)).isoformat()
    item["requested_at"] = late
    item["last_reinforced_at"] = late
    with pytest.raises(ValueError):
        AgentStateSnapshotV11.model_validate({**raw, "conversation_trace": future})
    texty = json.loads(json.dumps(raw["conversation_trace"]))
    texty["contexts"][0]["items"][0]["content"] = "what the user said"
    with pytest.raises(ValueError):
        AgentStateSnapshotV11.model_validate({**raw, "conversation_trace": texty})
    with pytest.raises(ValueError):
        AgentStateSnapshotV11.model_validate({**raw, "schema_version": 10})


def test_v11_projection_matches_an_independent_preflight() -> None:
    # Independent worst case: hand-built dicts, not TraceItem/ContextTrace.
    stamp = "9999-12-31T23:59:59.999999+00:00"
    per_context = MAX_TRACE_ITEMS_TOTAL // MAX_TRACE_CONTEXTS
    longest_float = 2.2250738585072014e-308
    worst = {
        "revision": MAX_TRACE_COUNTER,
        "contexts": [
            {
                "context_id": "c" * MAX_IDENTIFIER_CODEPOINTS,
                "items": [
                    {
                        "episode_id": "e" * MAX_IDENTIFIER_CODEPOINTS,
                        "turn_kind": "command",
                        "requested_at": stamp,
                        "event_sequence": MAX_PERSISTED_EVENT_SEQUENCE,
                        "surprise": longest_float,
                        "surprise_measured": True,
                        "stability_hours": longest_float,
                        "last_reinforced_at": stamp,
                        "last_reinforced_event_sequence": MAX_PERSISTED_EVENT_SEQUENCE,
                        "recall_count": MAX_TRACE_COUNTER,
                        "double_recall_count": MAX_TRACE_COUNTER,
                    }
                ]
                * per_context,
            }
        ]
        * MAX_TRACE_CONTEXTS,
    }
    preflight = len(
        json.dumps(worst, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )
    assert conversation_trace_schema_max_bytes() == preflight
    v10 = project_agent_state_v10_schema_max_bytes()
    v11 = project_agent_state_v11_schema_max_bytes()
    assert v11 == v10 + len('"conversation_trace":'.encode()) + preflight + len(b",")
    assert v11 <= AGENT_STATE_MAX_SERIALIZED_BYTES
    # The 16 MiB future-state reserve is inside the projection, untouched.
    assert v11 - AGENT_STATE_FUTURE_STATE_RESERVE_BYTES > v10 - AGENT_STATE_FUTURE_STATE_RESERVE_BYTES
