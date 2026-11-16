from pathlib import Path

import pytest

from suzka.chat_log import TITLE_MAX_CHARS, ChatLogStore, ChatSessionNotFound

from test_fastapi_backend import _client, _settings


def _chat_log_settings(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    return settings.model_copy(
        update={
            "chat_log": settings.chat_log.model_copy(
                update={"path": tmp_path / "private" / "chat_log" / "chat_log.sqlite3"}
            )
        }
    )


def test_store_keeps_turns_in_order_and_titles_from_the_first_message(tmp_path: Path) -> None:
    store = ChatLogStore(tmp_path / "log.sqlite3")
    session = store.create_session()
    store.append_exchange(session.session_id, "first   question\nhere", "first answer", "ep-1")
    store.append_exchange(session.session_id, "second", "second answer", "ep-2")

    turns = store.turns(session.session_id)
    assert [(t.role, t.content) for t in turns] == [
        ("user", "first   question\nhere"),
        ("assistant", "first answer"),
        ("user", "second"),
        ("assistant", "second answer"),
    ]
    assert turns[1].episode_id == "ep-1"
    assert store.get_session(session.session_id).title == "first question here"


def test_store_truncates_long_titles_and_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "log.sqlite3"
    session = ChatLogStore(path).create_session("x" * 200)
    reopened = ChatLogStore(path)
    assert len(reopened.get_session(session.session_id).title) == TITLE_MAX_CHARS


def test_store_delete_removes_turns_and_unknown_sessions_raise(tmp_path: Path) -> None:
    store = ChatLogStore(tmp_path / "log.sqlite3")
    session = store.create_session()
    store.append_exchange(session.session_id, "q", "a", None)
    store.delete_session(session.session_id)
    for call in (
        lambda: store.turns(session.session_id),
        lambda: store.append_exchange(session.session_id, "q", "a", None),
        lambda: store.delete_session(session.session_id),
        lambda: store.rename_session(session.session_id, "t"),
    ):
        with pytest.raises(ChatSessionNotFound):
            call()


def test_sessions_api_records_chat_turns_without_a_new_context(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_chat_log_settings(tmp_path)) as client:
        created = client.post("/api/chat/sessions", json={})
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        reply = client.post(
            "/api/chat",
            json={"message": "hello there", "attachments": [], "chat_session_id": session_id},
        )
        assert reply.status_code == 200
        assert reply.json()["transcript_saved"] is True

        detail = client.get(f"/api/chat/sessions/{session_id}").json()
        assert detail["session"]["title"] == "hello there"
        assert [(t["role"], t["content"]) for t in detail["turns"]] == [
            ("user", "hello there"),
            ("assistant", reply.json()["response"]),
        ]
        assert detail["turns"][1]["episode_id"] == reply.json()["episode_id"]

        # A transcript session is not a Context: the turn ran in the default one.
        registry = client.app.state.main_loop.context_registry
        assert registry.current_context_id == "conversation.default"
        assert [frame.context_id for frame in registry.state.frames] == ["conversation.default"]

        listed = client.get("/api/chat/sessions").json()["sessions"]
        assert [s["session_id"] for s in listed] == [session_id]

        renamed = client.patch(f"/api/chat/sessions/{session_id}", json={"title": "Greeting"})
        assert renamed.json()["title"] == "Greeting"
        assert client.delete(f"/api/chat/sessions/{session_id}").status_code == 204
        assert client.get(f"/api/chat/sessions/{session_id}").status_code == 404


def test_chat_without_a_session_reports_no_transcript(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_chat_log_settings(tmp_path)) as client:
        reply = client.post("/api/chat", json={"message": "plain", "attachments": []})
        assert reply.status_code == 200
        assert reply.json()["transcript_saved"] is None
        assert client.get("/api/chat/sessions").json()["sessions"] == []


def test_unknown_session_is_rejected_before_the_turn_runs(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_chat_log_settings(tmp_path)) as client:
        journal = client.app.state.event_journal
        before = len(journal.records)
        reply = client.post(
            "/api/chat",
            json={"message": "lost", "attachments": [], "chat_session_id": "missing-session"},
        )
        assert reply.status_code == 404
        assert len(journal.records) == before
