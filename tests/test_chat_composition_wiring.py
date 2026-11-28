"""#500 slice 3b-5: ``prompt_composition.mode`` wired into the chat turn.

Contract tested here: with ``legacy`` (default) every observable byte is the pre-change byte
(golden fixture captured from the unchanged code); a chat mode only changes the model INPUT, falls
back per turn to the legacy path when it cannot be used, never changes what is stored, and keeps
stored/user text out of the system role (#414).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from suzka.config.schema import PromptCompositionSettings
import suzka.persona.prompt_builder as prompt_builder_module
from suzka.persona.prompt_builder import PROMPT_HEADER, RUNTIME_DATA_HEADING
from suzka.runtime.main_loop import SuzkaMainLoop

from chat_composition_scenarios import (
    GOLDEN,
    MessagesProvider,
    PlainProvider,
    legacy_snapshot,
    normalize,
    normalized_writes,
    spy_trace_writes,
    stored,
)
from test_conversation_trace_chat import RoleProvider, _chat, _settings_with
from test_fastapi_backend import _client

HEADER = "\n".join(PROMPT_HEADER)
R15_LINE = (
    "Current interaction interpretation: UNKNOWN. A Context participant reference is not an "
    "authenticated speaker or person; do not infer trust, familiarity, reciprocity, caution, "
    "or interpersonal causation.\n\n"
)
LOGGER_NAME = "suzka.runtime.main_loop"
TURNS = ("my cat is called Mochi", "the weather is sunny today", "what is my cat called?")


def _settings(tmp_path: Path, mode: str = "chat_roles", *, trace: bool = True, k: int = 1, **limits: Any):  # type: ignore[no-untyped-def]
    base = _settings_with(tmp_path, enabled=trace)
    return base.model_copy(
        update={
            "memory": base.memory.model_copy(update={"db1_top_k": k}),
            "working_memory": base.working_memory.model_copy(update={"item_capacity": k}),
            "prompt_composition": PromptCompositionSettings(mode=mode, **limits),
        }
    )


def _labels(results: list[dict]) -> dict[str, str]:
    return {r["episode_id"]: f"<E{i}>" for i, r in enumerate(results)}


def _fallbacks(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "falls back to the legacy prompt" in r.getMessage()]


def _run(tmp_path: Path, settings, provider, messages=TURNS, **extra):  # type: ignore[no-untyped-def]
    with _client(tmp_path, settings=settings, provider=provider) as client:
        results = [_chat(client, message, **extra) for message in messages]
    return results


def _norm_calls(calls, results):  # type: ignore[no-untyped-def]
    labels = _labels(results)
    return [[normalize(s, labels), normalize(u, labels)] for s, u in calls]


# ------------------------------------------------------------------------ (a) golden, legacy


def test_legacy_default_is_byte_identical_to_the_pre_change_golden(tmp_path: Path) -> None:
    snapshot = json.loads(json.dumps(legacy_snapshot(tmp_path)))  # tuples -> lists, like the file
    assert snapshot == json.loads(GOLDEN.read_text())


def test_legacy_default_never_uses_generate_messages_even_if_available(tmp_path: Path) -> None:
    provider = MessagesProvider()
    results = _run(tmp_path, _settings_with(tmp_path, enabled=False), provider, TURNS[:2])
    golden = json.loads(GOLDEN.read_text())["attn_roles"]["calls"]
    assert provider.message_calls == []
    assert _norm_calls(provider.chat_calls, results)[0][0] == golden[0][0]
    assert len(provider.chat_calls) == 2


def test_private_interpretation_prefix_refactor_keeps_the_legacy_bytes() -> None:
    expected_prefix = (
        "Internal bounded first-person provisional interpretation, for the user's explicit "
        "request about the preceding exercise ONLY:\n"
        "- My earlier tentative option: A\n"
        "- My current tentative option: B\n"
        "- The earlier interpretation and the later report disagree.\n"
        "- Later user-reported correction exists: yes\n"
        "The user's later report is not independently verified human intent. "
        "Do not claim competence, personal identity, trust, objective success, "
        "or certainty. No hidden history or Source IDs are available. "
        "Use the current provisional option, not a stale earlier one.\n\n"
    )
    args = ("A", "B", 2, True, "CONTESTED")
    assert SuzkaMainLoop._private_interpretation_prefix(*args) == expected_prefix
    assert SuzkaMainLoop._prepend_private_interpretation("PROMPT", *args) == expected_prefix + "PROMPT"
    unknown = SuzkaMainLoop._prepend_private_interpretation("P", "A", "UNKNOWN", 2, True, "UNKNOWN")
    assert "- Current interpretation: undetermined.\n" in unknown and unknown.endswith("\n\nP")
    with pytest.raises(ValueError):
        SuzkaMainLoop._private_interpretation_prefix("A", "B", 1, True, "PRESENT")


# ----------------------------------------------------------------- (b) non-legacy happy path


@pytest.mark.parametrize("mode", ["chat_roles", "chat_roles_split"])
def test_chat_mode_sends_header_history_pairs_block_then_the_user_text(
    tmp_path: Path, mode: str
) -> None:
    provider = MessagesProvider()
    results = _run(tmp_path, _settings(tmp_path, mode), provider)
    assert provider.chat_calls == []  # the messages path replaced generate_chat
    last = provider.message_calls[-1]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user", "assistant", "user"]
    system = last[0]["content"]
    assert system.startswith(R15_LINE + HEADER)
    assert RUNTIME_DATA_HEADING not in system
    # History oldest -> newest; the stored-text block rides on the first (oldest) user turn.
    assert last[1]["content"].startswith(RUNTIME_DATA_HEADING + "\n")
    assert last[1]["content"].endswith("User message:\n" + TURNS[0])
    assert last[2]["content"] == "Visible API answer."
    assert last[3]["content"] == TURNS[1]
    assert last[4]["content"] == "Visible API answer."
    assert last[5]["content"] == TURNS[2]  # the final user message is the user text only
    # The legacy trace block is not repeated next to explicit history.
    assert "Recent conversation" not in "".join(m["content"] for m in last)
    if mode == "chat_roles_split":
        assert "Emotion:" in system  # code-origin sections follow the header in the system role
    else:
        assert "Emotion:" in last[1]["content"]
    # First turn: no history, so the block is on the only (final) user turn.
    first = provider.message_calls[0]
    assert [m["role"] for m in first] == ["system", "user"]
    assert first[1]["content"].endswith("User message:\n" + TURNS[0])
    assert len(results) == 3


# ------------------------------------------------------------------------ (c) #414 sentinels


class SentinelProvider(MessagesProvider):
    response_text = "Visible RESP-SENTINEL answer."


@pytest.mark.parametrize("mode", ["legacy", "chat_roles", "chat_roles_split"])
def test_sentinels_never_reach_the_system_message(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinels = ("USER-SENTINEL", "RESP-SENTINEL", "WM-SENTINEL", "SESSION-SENTINEL", "VALUE-SENTINEL")
    provider = SentinelProvider()
    settings = _settings(tmp_path, mode)
    with _client(tmp_path, settings=settings, provider=provider) as client:
        # ValuePromptEntry cannot be built outside ValueSystem; a Value name is stored-origin
        # text, so route a sentinel name through the builder's own values hook instead.
        def sentinel_values(add, _value_view):  # type: ignore[no-untyped-def]
            add("values", ["Active Values:", "- name=VALUE-SENTINEL"], "stored")

        monkeypatch.setattr(prompt_builder_module, "_add_values", sentinel_values)
        extra = {"client_session_id": "SESSION-SENTINEL"}
        _chat(client, "WM-SENTINEL first USER-SENTINEL", **extra)
        _chat(client, "second USER-SENTINEL", **extra)
        client.post(
            "/api/chat/debug",
            json={"message": "third USER-SENTINEL", "attachments": [], "debug": True, **extra},
        )
        _chat(client, "fourth USER-SENTINEL", **extra)
    systems = (
        [m[0]["content"] for m in provider.message_calls]
        if mode != "legacy"
        else [s for s, _u in provider.chat_calls]
    )
    assert len(systems) == 4
    for system in systems:
        for sentinel in sentinels:
            assert sentinel not in system, (mode, sentinel)
    # They are in the model input, just not in the system role (the test is not vacuous).
    everything = (
        json.dumps(provider.message_calls) if mode != "legacy" else json.dumps(provider.chat_calls)
    )
    for sentinel in ("USER-SENTINEL", "RESP-SENTINEL", "VALUE-SENTINEL"):
        assert sentinel in everything


# ------------------------------------------------------------------ (d) per-turn fallbacks


def _legacy_reference(tmp_path: Path, provider, trace: bool = True, **extra):  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "ref", "legacy", trace=trace)
    return _run(tmp_path / "ref", settings, provider, **extra)


def test_provider_without_messages_falls_back_to_legacy_bytes_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    reference = RoleProvider()
    ref_results = _legacy_reference(tmp_path, reference)
    provider = RoleProvider()  # generate_chat but no generate_messages
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = _run(tmp_path / "x", _settings(tmp_path / "x"), provider)
    assert _norm_calls(provider.chat_calls, results) == _norm_calls(reference.chat_calls, ref_results)
    [message] = _fallbacks(caplog)
    assert "provider_without_messages_or_chat_template" in message


def test_checkpoint_without_chat_template_falls_back_to_the_plain_prompt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class NoTemplate(MessagesProvider):
        def __init__(self) -> None:
            super().__init__()
            self.prompts: list[str] = []

        def uses_chat_roles(self) -> bool:
            return False

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return self.response_text

    reference = PlainProvider()
    ref_results = _legacy_reference(tmp_path, reference)
    provider = NoTemplate()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = _run(tmp_path / "x", _settings(tmp_path / "x"), provider)
    labels, ref_labels = _labels(results), _labels(ref_results)
    assert [normalize(p, labels) for p in provider.prompts] == [
        normalize(p, ref_labels) for p in reference.prompts
    ]
    assert provider.message_calls == [] and provider.chat_calls == []
    assert len(_fallbacks(caplog)) == 1


def test_participant_context_gets_no_history_and_the_legacy_bytes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    extra = {"client_session_id": "shared-session", "interlocutor_key": "ref-a"}
    reference = MessagesProvider()
    ref_results = _legacy_reference(tmp_path, reference, **extra)
    provider = MessagesProvider()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = _run(tmp_path / "x", _settings(tmp_path / "x"), provider, **extra)
    assert provider.message_calls == []
    assert _norm_calls(provider.chat_calls, results) == _norm_calls(reference.chat_calls, ref_results)
    [message] = _fallbacks(caplog)
    assert "participant_context" in message


def test_disabled_trace_falls_back_to_legacy_bytes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    reference = MessagesProvider()
    ref_results = _legacy_reference(tmp_path, reference, trace=False)
    provider = MessagesProvider()
    # The Settings validator refuses this combination at load time; the turn is still safe.
    settings = _settings(tmp_path / "x", trace=False)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = _run(tmp_path / "x", settings, provider)
    assert provider.message_calls == []
    assert _norm_calls(provider.chat_calls, results) == _norm_calls(reference.chat_calls, ref_results)
    [message] = _fallbacks(caplog)
    assert "conversation_trace_disabled" in message


def test_frozen_trace_falls_back_to_legacy(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with _client(tmp_path, settings=_settings(tmp_path, "legacy"), provider=MessagesProvider()) as client:
        _chat(client, "before freezing")
    provider = MessagesProvider()
    frozen = _settings(tmp_path, "chat_roles", trace=False)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        with _client(tmp_path, settings=frozen, provider=provider) as client:
            assert client.app.state.main_loop.conversation_trace.frozen
            _chat(client, "while frozen")
            _chat(client, "still frozen")
    assert provider.message_calls == [] and len(provider.chat_calls) == 2
    [message] = _fallbacks(caplog)
    assert "conversation_trace_frozen" in message


def test_decision_reasons_for_injected_builder_missing_attention_and_blank_input(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with _client(tmp_path, settings=_settings(tmp_path), provider=MessagesProvider()) as client:
        loop = client.app.state.main_loop
        context = SimpleNamespace(participant_refs=())
        event = payload = object()
        decide = loop._turn_composition
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert decide(event, context, payload, "hi") is loop.settings.prompt_composition
            assert decide(None, context, payload, "hi") is None  # no event
            assert decide(event, context, None, "hi") is None  # no Attention payload
            assert decide(event, context, payload, "  \n") is None
            assert decide(event, SimpleNamespace(participant_refs=("r",)), payload, "hi") is None
            real_builder = loop.prompt_builder
            loop.prompt_builder = SimpleNamespace(build=real_builder.build)
            assert decide(event, context, payload, "hi") is None
            assert decide(event, context, payload, "hi") is None  # second time: no second warning
            loop.prompt_builder = real_builder
    reasons = [m.split("(", 1)[1].split(")", 1)[0] for m in _fallbacks(caplog)]
    assert reasons == [
        "no_attention_payload",
        "blank_user_input",
        "participant_context",
        "custom_prompt_builder",
    ]


# ------------------------------------------------------------- (e) stored data is unchanged


def test_stored_episode_response_and_session_turn_are_identical_across_modes(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    for name, mode in (("legacy", "legacy"), ("roles", "chat_roles"), ("split", "chat_roles_split")):
        provider = MessagesProvider() if mode != "legacy" else RoleProvider()
        with _client(tmp_path / name, settings=_settings(tmp_path / name, mode), provider=provider) as client:
            results = [_chat(client, message) for message in TURNS]
            seen[name] = {
                "stored": [stored(client, r["episode_id"]) for r in results],
                "responses": [r["response"] for r in results],
                "session": client.app.state.main_loop.session_state.context_text(),
            }
    assert seen["legacy"]["stored"] == [
        {"user_input": message, "response": "Visible API answer."} for message in TURNS
    ]
    assert seen["roles"] == seen["legacy"] == seen["split"]


# ---------------------------------------------------------------------- (f) reinforce mapping


def _writes(tmp_path: Path, settings, provider=None, messages=TURNS):  # type: ignore[no-untyped-def]
    with _client(tmp_path, settings=settings, provider=provider or MessagesProvider()) as client:
        calls = spy_trace_writes(client)
        results = [_chat(client, message) for message in messages]
    return normalized_writes(calls, _labels(results))


def _reinforces(writes: list[dict]) -> list[dict]:
    return [w for w in writes if w["call"] == "reinforce"]


def test_reinforce_uses_the_exchanges_that_were_sent_as_history(tmp_path: Path) -> None:
    writes = _writes(tmp_path, _settings(tmp_path))
    assert [w["call"] for w in writes] == ["reinforce", "add"] * 3  # no new writes
    first, second, third = _reinforces(writes)
    assert first["shown"] == [] and second["shown"] == ["<E0>"]
    assert third["shown"] == ["<E0>", "<E1>"]  # both were sent as history
    assert set(third["double"]) <= {"<E0>", "<E1>"} and third["double"]


def test_reinforce_follows_a_window_smaller_than_the_history(tmp_path: Path) -> None:
    writes = _writes(tmp_path / "w", _settings(tmp_path / "w", history_window=1))
    assert _reinforces(writes)[2]["shown"] == ["<E1>"]  # only the newest was sent
    # A byte cap that fits one exchange but not two (window itself is 6).
    writes = _writes(tmp_path / "b", _settings(tmp_path / "b", max_history_bytes=70))
    assert _reinforces(writes)[2]["shown"] == ["<E1>"]


def test_reinforce_keeps_the_legacy_trace_when_no_history_is_sent(tmp_path: Path) -> None:
    golden = json.loads(GOLDEN.read_text())["trace_roles"]["writes"]
    # Caps drop all history: the trace block stays in the message, so the legacy view is kept.
    writes = _writes(tmp_path, _settings(tmp_path, max_history_bytes=0))
    assert writes == golden


# ------------------------------------------------------- (g) evidence limit numbers and overcount


def _wm_rows(text: str) -> int:
    return text.count('{"source_kind"')


def test_evidence_limit_cuts_the_model_input_and_not_the_overcounted_double_recall(
    tmp_path: Path,
) -> None:
    from suzka.runtime.main_loop import SuzkaMainLoop as Loop

    settings = _settings(tmp_path, k=3, history_window=1, evidence_top_k=1, evidence_item_max_chars=30)
    provider = MessagesProvider()
    captured: list[tuple[Any, Any, Any, frozenset[str]]] = []
    real = Loop._episodes_in_working_memory_prompt

    def spy(view, payload, row_limit=None):  # type: ignore[no-untyped-def]
        result = real(view, payload, row_limit)
        captured.append((view, payload, row_limit, result))
        return result

    Loop._episodes_in_working_memory_prompt = staticmethod(spy)  # type: ignore[method-assign]
    try:
        with _client(tmp_path, settings=settings, provider=provider) as client:
            calls = spy_trace_writes(client)
            results = [_chat(client, m) for m in (*TURNS, "and what is the weather?")]
            debug = client.post(
                "/api/chat/debug", json={"message": "dbg", "attachments": [], "debug": True}
            ).json()
    finally:
        Loop._episodes_in_working_memory_prompt = staticmethod(real)  # type: ignore[method-assign]
    # The model input holds one evidence row, cut to 30 characters (ellipsis counted).
    blocks = [m["content"] for call in provider.message_calls for m in call if RUNTIME_DATA_HEADING in m["content"]]
    assert all(_wm_rows(b) <= 1 for b in blocks)
    assert any(_wm_rows(b) == 1 for b in blocks)
    # (view, payload, limit, result) of the last real chat turn that had rows to drop.
    dropped = [c for c in captured if c[2] is not None and c[2][0] > c[2][1]]
    assert dropped
    view, payload, limit, limited = dropped[-1]
    full = real(view, payload, None)
    assert limited < full  # rows the limit dropped are not counted as shown
    assert len(limited) <= limit[1]
    # The same rows are what the legacy counting would have added (the overcount being fixed).
    assert full - limited
    # Debug carries the numbers only (no text).
    numbers = debug["evidence_limit"]
    assert numbers["skipped"] is None and numbers["rows_after"] <= 1
    assert set(numbers) == {
        "rows_before", "rows_after", "chars_before", "chars_after",
        "evidence_chars_before", "evidence_chars_after", "skipped",
    }
    last_double = _reinforces(normalized_writes(calls, _labels(results)))[-1]["double"]
    assert all(isinstance(item, str) for item in last_double)


def test_row_limit_helper_fails_closed_on_a_row_count_mismatch() -> None:
    from suzka.runtime.main_loop import SuzkaMainLoop as Loop

    view = SimpleNamespace(selected=())
    payload = SimpleNamespace(included_candidate_ids=())
    # No Working Memory rows at all but the limit claims 2: nothing can count, nothing breaks.
    assert Loop._episodes_in_working_memory_prompt(view, payload, (2, 1)) == frozenset()  # type: ignore[arg-type]


# --------------------------------------------------------------------- (h) debug and the API


def test_debug_turn_has_optional_fields_only_in_a_chat_mode_and_stays_read_only(tmp_path: Path) -> None:
    provider = MessagesProvider()
    with _client(tmp_path, settings=_settings(tmp_path), provider=provider) as client:
        for message in TURNS[:2]:
            _chat(client, message)
        trace = client.app.state.main_loop.conversation_trace
        before = trace.state
        response = client.post(
            "/api/chat/debug", json={"message": "dbg turn", "attachments": [], "debug": True}
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert trace.state == before  # no forget/drop/reinforce/add
    sent = provider.message_calls[-1]
    assert data["composed_messages"] == sent
    assert data["composed_messages"][-1] == {"role": "user", "content": "dbg turn"}
    report = data["composition_report"]
    assert report["mode"] == "chat_roles" and report["exchanges_used"] == 2
    assert data["evidence_limit"]["skipped"] is None
    # `prompt` stays the legacy joined string (header, runtime data, User:/Assistant: tail).
    assert data["prompt"].startswith(HEADER) and data["prompt"].endswith("User: dbg turn\nAssistant:")
    assert "Recent conversation" in data["prompt"]  # legacy trace block, as before
    items = {i["episode_id"]: i for i in data["conversation_trace"]["items"]}
    assert {i["display"] for i in items.values()} <= {"history", "in_working_memory", "hidden_by_budget"}


def test_legacy_debug_turn_leaves_the_optional_fields_empty(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_settings(tmp_path, "legacy"), provider=RoleProvider()) as client:
        _chat(client, "hello")
        data = client.post(
            "/api/chat/debug", json={"message": "dbg", "attachments": [], "debug": True}
        ).json()
    assert data["composed_messages"] is None
    assert data["composition_report"] is None
    assert data["evidence_limit"] is None
    legacy = json.loads(GOLDEN.read_text())["attn_roles"]["debug_keys"]
    assert sorted(k for k in data if k not in {"composed_messages", "composition_report", "evidence_limit"}) == legacy


def test_private_interpretation_and_r15_lines_lead_the_system_message_in_legacy_order(
    tmp_path: Path,
) -> None:
    prefix = SuzkaMainLoop._private_interpretation_prefix("A", "B", 2, True, "CONTESTED")
    fake = SimpleNamespace(
        qualified=True, original_referent="A", current_referent="B", revision=2,
        later_user_reported=True, status_class="CONTESTED",
    )
    systems: dict[str, str] = {}
    for mode in ("legacy", "chat_roles"):
        provider = MessagesProvider()
        with _client(tmp_path / mode, settings=_settings(tmp_path / mode, mode), provider=provider) as client:
            loop = client.app.state.main_loop
            loop._requests_prior_interpretation = lambda _text: True  # type: ignore[method-assign]
            loop._r15_state_port._read_private_narrative_in_chat = lambda *_a: fake  # type: ignore[method-assign]
            _chat(client, "earlier option?")
        systems[mode] = (
            provider.chat_calls[0][0] if mode == "legacy" else provider.message_calls[0][0]["content"]
        )
    assert systems["legacy"] == systems["chat_roles"] == prefix + R15_LINE + HEADER
