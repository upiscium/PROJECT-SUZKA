"""#417 PR-C: Chat adds, shows and reads the ConversationTrace end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from suzka.runtime.conversation_trace import (
    COMMAND_TURN_TEXT,
    RECENT_CONVERSATION_HEADING,
    TraceTurnKind,
)

from test_fastapi_backend import ThinkingProvider, _client, _settings

CTX = "conversation.default"
EXPERIENCE = "experience-00000000-0000-4000-8000-000000000001"


class RoleProvider(ThinkingProvider):
    """Records what goes to each chat role."""

    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, str]] = []

    def uses_chat_roles(self) -> bool:
        return True

    def generate_chat(self, system_text: str, user_text: str) -> str:
        self.chat_calls.append((system_text, user_text))
        return self.response_text


def _settings_with(tmp_path: Path, *, enabled: bool):  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    return settings.model_copy(
        update={
            "conversation_trace": settings.conversation_trace.model_copy(
                update={"enabled": enabled}
            )
        }
    )


def _chat(client, message: str, **extra: object) -> dict:  # type: ignore[no-untyped-def]
    response = client.post("/api/chat", json={"message": message, "attachments": [], **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _one_retrieval(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Enabled trace; retrieval and Working Memory hold one episode, so WM shows one past turn."""

    settings = _settings_with(tmp_path, enabled=True)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(update={"db1_top_k": 1}),
            "working_memory": settings.working_memory.model_copy(update={"item_capacity": 1}),
        }
    )


def test_earlier_turns_are_remembered_once_in_the_user_role(tmp_path: Path) -> None:
    provider = RoleProvider()
    with _client(tmp_path, settings=_one_retrieval(tmp_path), provider=provider) as client:
        first = _chat(client, "my cat is called Mochi")
        trace = client.app.state.main_loop.conversation_trace
        [item] = trace.state.items_for(CTX)
        assert item.episode_id == first["episode_id"]
        assert item.turn_kind is TraceTurnKind.CHAT
        _chat(client, "the weather is sunny today")
        _chat(client, "what is my cat called?")
    system_text, user_text = provider.chat_calls[-1]
    assert RECENT_CONVERSATION_HEADING not in system_text
    assert RECENT_CONVERSATION_HEADING in user_text
    # Each earlier turn reaches the model exactly once: through Working Memory
    # (one retrieved episode) or through the trace block, never both.
    for message in ("my cat is called Mochi", "the weather is sunny today"):
        assert user_text.count(message) == 1, message
    block = user_text.split(RECENT_CONVERSATION_HEADING, 1)[1].split("User message:")[0]
    assert '— User: "' in block
    # The first turn had nothing to remember.
    assert RECENT_CONVERSATION_HEADING not in provider.chat_calls[0][1]


def test_disabled_by_default_changes_nothing(tmp_path: Path) -> None:
    provider = RoleProvider()
    with _client(tmp_path, provider=provider) as client:
        _chat(client, "hello")
        _chat(client, "again")
        assert client.app.state.main_loop.conversation_trace.state.contexts == ()
    assert all(RECENT_CONVERSATION_HEADING not in user for _system, user in provider.chat_calls)


def test_debug_chat_reads_the_trace_without_changing_it(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_one_retrieval(tmp_path)) as client:
        first = _chat(client, "remember this")
        second = _chat(client, "and this as well")
        trace = client.app.state.main_loop.conversation_trace
        before = trace.state
        response = client.post(
            "/api/chat/debug",
            json={"message": "debug turn", "attachments": [], "debug": True},
        )
        assert response.status_code == 200, response.text
        data = response.json()["conversation_trace"]
        assert data["state"] == "enabled"
        items = {item["episode_id"]: item for item in data["items"]}
        assert set(items) == {first["episode_id"], second["episode_id"]}
        assert all(item["read_status"] == "resolved" for item in items.values())
        displays = sorted(item["display"] for item in items.values())
        # One past turn comes through Working Memory (deduped), one through the trace.
        assert displays == ["in_working_memory", "shown"]
        in_wm = next(item for item in items.values() if item["display"] == "in_working_memory")
        assert in_wm["double_recall"] is True
        assert trace.state == before


def test_a_failed_turn_leaves_the_trace_as_committed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import suzka.runtime.main_loop as main_loop_module

    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        _chat(client, "kept")
        trace = client.app.state.main_loop.conversation_trace
        committed = trace.state

        def fail(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("failure after the trace was updated")

        monkeypatch.setattr(main_loop_module, "SessionTurnParticipant", fail)
        with pytest.raises(Exception):
            client.post("/api/chat", json={"message": "lost", "attachments": []})
        monkeypatch.undo()
        assert trace.state == committed
        _chat(client, "works again")
        assert len(trace.state.items_for(CTX)) == 2


def test_a_missing_episode_is_kept_hidden_and_logged_once(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime, timedelta, timezone

    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        _chat(client, "first")
        trace = client.app.state.main_loop.conversation_trace
        sequence = client.app.state.agent_state_store.load().last_processed_event_sequence
        trace.add(CTX, episode_id="episode-that-does-not-exist", turn_kind=TraceTurnKind.CHAT,
                  requested_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                  event_sequence=sequence, surprise=0.0)
        with caplog.at_level("WARNING", logger="suzka.runtime.main_loop"):
            _chat(client, "second")
            _chat(client, "third")
        defects = [r for r in caplog.records if "episode-that-does-not-exist is missing" in r.getMessage()]
        assert len(defects) == 1
        episode_ids = [item.episode_id for item in trace.state.items_for(CTX)]
        assert "episode-that-does-not-exist" in episode_ids
        response = client.post(
            "/api/chat/debug",
            json={"message": "debug", "attachments": [], "debug": True},
        )
        missing = [i for i in response.json()["conversation_trace"]["items"]
                   if i["episode_id"] == "episode-that-does-not-exist"]
        assert missing and missing[0]["read_status"] == "missing"
        assert missing[0]["display"] == "unreadable"


def test_command_turns_are_remembered_as_a_fixed_note(tmp_path: Path) -> None:
    provider = RoleProvider()
    with _client(tmp_path, settings=_one_retrieval(tmp_path), provider=provider) as client:
        _chat(client, f"REVIEW_EXPERIENCE:{EXPERIENCE}")
        [item] = client.app.state.main_loop.conversation_trace.state.items_for(CTX)
        assert item.turn_kind is TraceTurnKind.COMMAND
        _chat(client, "an ordinary turn")
        _chat(client, "next")
    _system, user_text = provider.chat_calls[-1]
    if RECENT_CONVERSATION_HEADING in user_text:
        block = user_text.split(RECENT_CONVERSATION_HEADING, 1)[1].split("User message:")[0]
        assert EXPERIENCE not in block
        if "REVIEW_EXPERIENCE" not in user_text:
            # Not shown through Working Memory, so the trace shows the note.
            assert COMMAND_TURN_TEXT in block


def test_contexts_with_participants_do_not_use_the_trace(tmp_path: Path) -> None:
    provider = RoleProvider()
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True), provider=provider) as client:
        selectors = {"client_session_id": "shared-session", "interlocutor_key": "ref-a"}
        _chat(client, "secret from ref-a", **selectors)
        _chat(client, "second", **selectors)
        assert client.app.state.main_loop.conversation_trace.state.contexts == ()
    # No trace block (earlier text may still arrive through DB1 retrieval).
    assert all(RECENT_CONVERSATION_HEADING not in user for _system, user in provider.chat_calls)


def test_a_frozen_trace_is_neither_shown_nor_changed(tmp_path: Path) -> None:
    settings = _settings_with(tmp_path, enabled=True)
    with _client(tmp_path, settings=settings) as client:
        _chat(client, "before freezing")
        saved = client.app.state.main_loop.conversation_trace.state
    disabled = _settings_with(tmp_path, enabled=False)
    provider = RoleProvider()
    with _client(tmp_path, settings=disabled, provider=provider) as client:
        trace = client.app.state.main_loop.conversation_trace
        assert trace.frozen
        _chat(client, "while frozen")
        assert trace.state == saved
    assert all(RECENT_CONVERSATION_HEADING not in user for _system, user in provider.chat_calls)
