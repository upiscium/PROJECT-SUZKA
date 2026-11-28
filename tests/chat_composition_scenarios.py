"""Shared scenario runners for the #500 slice 3b-5 wiring tests (not a test module).

``legacy_snapshot`` runs real FastAPI chat turns and returns everything observable to the
model, the store and the debug API, normalized only where the clock decides the value (the
emotion ``arousal``/``optimal_loss`` numbers, episode ids, retention numbers).  The checked-in
fixture ``golden/chat_composition_legacy.json`` was captured from the code BEFORE the wiring
change, so the golden test proves byte-identity of the default (legacy) path.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from test_conversation_trace_chat import RoleProvider, _chat, _one_retrieval
from test_fastapi_backend import ThinkingProvider, _client

GOLDEN = Path(__file__).parent / "golden" / "chat_composition_legacy.json"


class PlainProvider(ThinkingProvider):
    """No chat roles at all: only ``generate(prompt)``."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


class MessagesProvider(RoleProvider):
    """Has both ``generate_chat`` and ``generate_messages`` (records both)."""

    def __init__(self) -> None:
        super().__init__()
        self.message_calls: list[list[dict[str, str]]] = []

    def generate_messages(self, messages):  # type: ignore[no-untyped-def]
        self.message_calls.append([dict(message) for message in messages])
        return self.response_text


_CLOCK_NUMBERS = re.compile(r"(arousal|optimal_loss|valence): -?[0-9]+\.[0-9]+")


def normalize(text: str, episode_ids: dict[str, str] | None = None) -> str:
    text = _CLOCK_NUMBERS.sub(lambda m: f"{m.group(1)}: <N>", text)
    for real, label in (episode_ids or {}).items():
        text = text.replace(real, label)
    return text


def spy_trace_writes(client: Any) -> list[tuple[str, dict[str, Any]]]:
    """Record every ``reinforce``/``add`` on the live ConversationTrace (still calling it)."""

    trace = client.app.state.main_loop.conversation_trace
    calls: list[tuple[str, dict[str, Any]]] = []
    for name in ("reinforce", "add"):
        original = getattr(trace, name)

        def make(original=original, name=name):  # type: ignore[no-untyped-def]
            def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
                calls.append((name, {"args": args, "kwargs": dict(kwargs)}))
                return original(*args, **kwargs)

            return spy

        setattr(trace, name, make())
    return calls


def normalized_writes(
    calls: list[tuple[str, dict[str, Any]]], labels: dict[str, str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, payload in calls:
        kwargs = payload["kwargs"]
        if name == "reinforce":
            out.append({
                "call": "reinforce",
                "context_id": payload["args"][0],
                "shown": sorted(labels.get(i, i) for i in kwargs["shown_episode_ids"]),
                "double": sorted(labels.get(i, i) for i in kwargs["double_recall_episode_ids"]),
            })
        else:
            out.append({
                "call": "add",
                "context_id": payload["args"][0],
                "episode": labels.get(kwargs["episode_id"], kwargs["episode_id"]),
                "turn_kind": kwargs["turn_kind"].value,
            })
    return out


def stored(client: Any, episode_id: str) -> dict[str, str]:
    record = client.app.state.main_loop.memory_system.get_committed_episodic(episode_id).record
    return {"user_input": record.user_input, "response": record.response}


def legacy_snapshot(tmp_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # Attention path, trace off, chat-roles provider (system, user), plus a debug turn.
    provider = RoleProvider()
    with _client(tmp_path / "attn", provider=provider) as client:
        first = _chat(client, "my cat is called Mochi")
        second = _chat(client, "what is my cat called?")
        debug = client.post(
            "/api/chat/debug", json={"message": "dbg", "attachments": [], "debug": True}
        ).json()
        labels = {first["episode_id"]: "<E0>", second["episode_id"]: "<E1>"}
        out["attn_roles"] = {
            "calls": [[normalize(s, labels), normalize(u, labels)] for s, u in provider.chat_calls],
            "debug_prompt": normalize(debug["prompt"], labels),
            "debug_keys": sorted(
                key for key in debug if key not in {"composed_messages", "composition_report", "evidence_limit"}
            ),
            "stored": [stored(client, first["episode_id"]), stored(client, second["episode_id"])],
            "session": client.app.state.main_loop.session_state.context_text(),
            "responses": [first["response"], second["response"]],
        }

    # Attention path, provider without chat roles: the single built prompt string.
    plain = PlainProvider()
    with _client(tmp_path / "plain", provider=plain) as client:
        _chat(client, "my cat is called Mochi")
        _chat(client, "what is my cat called?")
    out["attn_plain"] = [normalize(prompt) for prompt in plain.prompts]

    # Trace on: trace block in the user message, reinforce/add calls, debug diagnostics.
    traced = RoleProvider()
    with _client(tmp_path / "trace", settings=_one_retrieval(tmp_path / "trace"), provider=traced) as client:
        calls = spy_trace_writes(client)
        results = [
            _chat(client, "my cat is called Mochi"),
            _chat(client, "the weather is sunny today"),
            _chat(client, "what is my cat called?"),
        ]
        debug = client.post(
            "/api/chat/debug", json={"message": "dbg", "attachments": [], "debug": True}
        ).json()
        labels = {r["episode_id"]: f"<E{i}>" for i, r in enumerate(results)}
        out["trace_roles"] = {
            "calls": [[normalize(s, labels), normalize(u, labels)] for s, u in traced.chat_calls],
            "debug_prompt": normalize(debug["prompt"], labels),
            "debug_trace": {
                "state": debug["conversation_trace"]["state"],
                "items": sorted(
                    (labels[i["episode_id"]], i["display"], i["double_recall"], i["read_status"])
                    for i in debug["conversation_trace"]["items"]
                ),
            },
            "writes": normalized_writes(calls, labels),
            "stored": [stored(client, r["episode_id"]) for r in results],
            "session": client.app.state.main_loop.session_state.context_text(),
        }
    return out
