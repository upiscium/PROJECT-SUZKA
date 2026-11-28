"""#500 slice 3b-3: the history of recent exchanges is exposed from the trace view, read-only."""

from __future__ import annotations

from pathlib import Path

from suzka.runtime.conversation_trace import render_trace

from test_conversation_trace_chat import (
    RoleProvider,
    _chat,
    _one_retrieval,
    _settings_with,
)
from test_fastapi_backend import _client


def _spy_views(client):  # type: ignore[no-untyped-def]
    loop = client.app.state.main_loop
    views: list = []
    original = loop._conversation_trace_view

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        view = original(*args, **kwargs)
        views.append(view)
        return view

    loop._conversation_trace_view = spy
    return views


def test_history_includes_working_memory_episodes_and_keeps_legacy_trace(tmp_path: Path) -> None:
    provider = RoleProvider()
    with _client(tmp_path, settings=_one_retrieval(tmp_path), provider=provider) as client:
        first = _chat(client, "my cat is called Mochi")
        second = _chat(client, "the weather is sunny today")
        views = _spy_views(client)
        loop = client.app.state.main_loop
        _chat(client, "what is my cat called?")
    [view] = views
    history = view.history(6)
    assert [entry.episode_id for entry in history] == [first["episode_id"], second["episode_id"]]
    assert history[0].user_text == "my cat is called Mochi"
    assert history[0].assistant_text == first["response"]
    # One of the two turns is shown through Working Memory, so the legacy
    # trace omits it but the history keeps it (no hole in the chronology).
    assert view.in_working_memory_ids
    assert len(view.rendered.shown_episode_ids) == 1
    omitted = {e.episode_id for e in history} - set(view.rendered.shown_episode_ids)
    assert omitted == set(view.in_working_memory_ids)
    # The legacy rendering is exactly render_trace over the readable turns
    # that Working Memory does not show, and that text is what the model got.
    legacy = [t for t in view.readable_turns if t.item.episode_id not in view.in_working_memory_ids]
    assert view.rendered == render_trace(legacy, view.now, loop.conversation_trace.parameters)
    assert view.rendered.text and view.rendered.text in provider.chat_calls[-1][1]


def test_history_leaves_out_unreadable_episodes(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from suzka.runtime.conversation_trace import TraceTurnKind

    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        first = _chat(client, "first")
        loop = client.app.state.main_loop
        sequence = client.app.state.agent_state_store.load().last_processed_event_sequence
        loop.conversation_trace.add(
            "conversation.default",
            episode_id="episode-that-does-not-exist",
            turn_kind=TraceTurnKind.CHAT,
            requested_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            event_sequence=sequence,
            surprise=0.0,
        )
        views = _spy_views(client)
        _chat(client, "second")
    [view] = views
    assert [e.episode_id for e in view.history(6)] == [first["episode_id"]]


def test_history_is_empty_without_a_trace_view(tmp_path: Path) -> None:
    # participant Context
    with _client(tmp_path, settings=_settings_with(tmp_path, enabled=True)) as client:
        views = _spy_views(client)
        selectors = {"client_session_id": "shared-session", "interlocutor_key": "ref-a"}
        _chat(client, "secret", **selectors)
        _chat(client, "second", **selectors)
        assert views == [None, None]
    # frozen
    frozen_dir = tmp_path / "f"
    with _client(frozen_dir, settings=_settings_with(frozen_dir, enabled=True)) as client:
        _chat(client, "before freezing")
    with _client(frozen_dir, settings=_settings_with(frozen_dir, enabled=False)) as client:
        views = _spy_views(client)
        _chat(client, "while frozen")
        assert views == [None]


def test_debug_chat_history_is_read_only(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_one_retrieval(tmp_path)) as client:
        first = _chat(client, "remember this")
        _chat(client, "and this as well")
        loop = client.app.state.main_loop
        trace = loop.conversation_trace
        before = trace.state
        writes: list[str] = []
        for name in (
            "save_episodic",
            "save_semantic",
            "save_legacy_semantic",
            "publish_coordinated_episodic",
            "project_semantic_revision",
        ):

            def record(*a: object, _n: str = name, _o: object = getattr(loop.memory_system, name), **k: object) -> object:
                writes.append(_n)
                return _o(*a, **k)  # type: ignore[operator]

            setattr(loop.memory_system, name, record)
        views = _spy_views(client)
        response = client.post(
            "/api/chat/debug", json={"message": "debug", "attachments": [], "debug": True}
        )
        assert response.status_code == 200, response.text
        [view] = views
        # The debug turn itself is an ordinary committed chat (its own episode
        # write is existing behaviour). Exposing the history adds no write and
        # leaves the trace as the debug read left it (never reinforced/dropped).
        assert trace.state == before
        writes.clear()
        assert first["episode_id"] in [e.episode_id for e in view.history(6)]
        assert writes == []
        assert trace.state == before


def test_legacy_rendered_trace_is_pinned_to_literal_text() -> None:
    """Golden: literals captured from the unchanged render_trace (#500 3b-3 review)."""

    from datetime import datetime, timedelta, timezone

    from suzka.runtime.conversation_trace import (
        RecalledTurn,
        TraceItem,
        TraceParameters,
        TraceTurnKind,
        recent_history,
    )

    t0 = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)

    def turn(n: int, minutes: int, kind: TraceTurnKind, retention: float, user: str, reply: str):  # type: ignore[no-untyped-def]
        at = t0 + timedelta(minutes=minutes)
        item = TraceItem(
            episode_id=f"episode-g-{n:04d}", turn_kind=kind, requested_at=at, event_sequence=n,
            surprise=0.0, surprise_measured=True, stability_hours=1.0,
            last_reinforced_at=at, last_reinforced_event_sequence=n,
        )
        return RecalledTurn(item, retention, user, reply)

    faint = turn(1, 0, TraceTurnKind.CHAT, 0.2,
                 "my cat is called Mochi and she likes the window in the afternoon sun a lot",
                 "Mochi sounds lovely, I will remember that she likes the window seat")
    command = turn(2, 10, TraceTurnKind.COMMAND, 0.9, "REVIEW", "x")
    in_wm = turn(3, 20, TraceTurnKind.CHAT, 0.9, "what is\nthe weather", "it is sunny")
    now = t0 + timedelta(minutes=30)

    # Episode 3 is shown through Working Memory, so the legacy trace omits it.
    rendered = render_trace([faint, command], now, TraceParameters())
    assert rendered.text == (
        "Recent conversation (as you remember it; faint items are shortened):\n"
        '- about 30 min ago (faint) — User: "my cat is called Mochi and she likes the window'
        ' in the after…" / You: "Mochi sounds lovely, I will remember that she likes the'
        ' wind…"\n'
        "- about 20 min ago — (the user asked to review an earlier turn)"
    )
    assert rendered.shown_episode_ids == ("episode-g-0001", "episode-g-0002")
    assert rendered.faint_episode_ids == ("episode-g-0001",)
    # The history still holds the WM-shown episode, with raw text.
    history = recent_history([faint, command, in_wm], 6)
    assert [(e.episode_id, e.user_text, e.assistant_text) for e in history] == [
        ("episode-g-0001", faint.user_text, faint.response_text),
        ("episode-g-0003", "what is\nthe weather", "it is sunny"),
    ]


def test_legacy_diagnostics_are_pinned_to_literal_values(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_one_retrieval(tmp_path)) as client:
        _chat(client, "remember this")
        _chat(client, "and this as well")
        response = client.post(
            "/api/chat/debug", json={"message": "debug", "attachments": [], "debug": True}
        )
        items = response.json()["conversation_trace"]["items"]
    assert response.json()["conversation_trace"]["state"] == "enabled"
    assert sorted((i["display"], i["read_status"], i["double_recall"]) for i in items) == [
        ("in_working_memory", "resolved", True),
        ("shown", "resolved", False),
    ]
