"""#417 PR-C: Chat adds, shows and reads the ConversationTrace end to end."""

from __future__ import annotations

from pathlib import Path

from suzka.runtime.conversation_trace import (
    COMMAND_TURN_TEXT,
    RECENT_CONVERSATION_HEADING,
    TraceTurnKind,
)

from test_fastapi_backend import ThinkingProvider, _client, _settings, admin_headers

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


def test_the_previous_turn_is_remembered_in_the_user_role(tmp_path: Path) -> None:
    provider = RoleProvider()
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True), provider=provider) as client:
        first = _chat(client, "my cat is called Mochi")
        trace = client.app.state.main_loop.conversation_trace
        [item] = trace.state.items_for(CTX)
        assert item.episode_id == first["episode_id"]
        assert item.turn_kind is TraceTurnKind.CHAT
        _chat(client, "what is my cat called?")
    system_text, user_text = provider.chat_calls[-1]
    assert RECENT_CONVERSATION_HEADING not in system_text
    assert RECENT_CONVERSATION_HEADING in user_text
    assert "just now — User: my cat is called Mochi / You: Visible API answer." in user_text
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
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        first = _chat(client, "remember this")
        trace = client.app.state.main_loop.conversation_trace
        before = trace.state
        response = client.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={"message": "debug turn", "attachments": [], "debug": True},
        )
        assert response.status_code == 200, response.text
        data = response.json()["conversation_trace"]
        assert data["state"] == "enabled"
        assert [item["episode_id"] for item in data["items"]] == [first["episode_id"]]
        assert data["items"][0]["read_status"] == "resolved"
        assert data["items"][0]["display"] == "shown"
        assert trace.state == before


def test_command_turns_are_remembered_as_a_fixed_note(tmp_path: Path) -> None:
    provider = RoleProvider()
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True), provider=provider) as client:
        _chat(client, f"REVIEW_EXPERIENCE:{EXPERIENCE}")
        [item] = client.app.state.main_loop.conversation_trace.state.items_for(CTX)
        assert item.turn_kind is TraceTurnKind.COMMAND
        _chat(client, "next")
    _system, user_text = provider.chat_calls[-1]
    assert COMMAND_TURN_TEXT in user_text
    assert EXPERIENCE not in user_text.split(RECENT_CONVERSATION_HEADING, 1)[1].split("User message:")[0]


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
