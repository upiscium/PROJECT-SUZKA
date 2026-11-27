"""Legacy equality with split_chat_prompt for prompts that carry an Attention payload (real builder
output). Reuses the payload helpers of test_attention_prompt, as other prompt tests do."""

from __future__ import annotations

import pytest
from test_attention_prompt import _capture, _payload

from suzka.body import EmotionState
from suzka.identity import ValuePromptView
from suzka.persona.chat_composition import ChatMessage, compose, composition_input_from_split
from suzka.persona.prompt_builder import ContextPromptView, PromptBuilder, split_chat_prompt
from suzka.working_memory_contracts import WorkingMemoryView


@pytest.mark.parametrize("context", [True, False])
@pytest.mark.parametrize("trace", ["", "Recent conversation:\n- user: hi\n- assistant: hello"])
@pytest.mark.parametrize("message", ["hello", "その都市の人口は", "User message:\nx"])
def test_legacy_equals_split_chat_prompt_with_attention_payload(context: bool, trace: str, message: str) -> None:
    items, view, motivation, goals, commitments, refresh = _capture()
    payload = _payload(items, view, motivation, goals, commitments, refresh)
    frame = ContextPromptView(
        context_id="conversation-default", context_type="conversation", source_channel="chat",
        source_session_id=None, participant_refs=(),
    )
    empty = WorkingMemoryView(
        selected=(), decisions=(), projected_bytes=0, item_capacity=8, projection_max_bytes=1024, revision=0,
    )
    prompt = PromptBuilder().build(
        message, EmotionState(), empty, frame if context else None, ValuePromptView(()),
        attention_payload=payload, conversation_trace_text=trace,
    )
    split = split_chat_prompt(prompt, message)
    assert split is not None
    system, user = split
    inp = composition_input_from_split("Neutral line.\n" + system, user, message)
    assert inp is not None
    assert compose("legacy", inp) == (ChatMessage("system", "Neutral line.\n" + system), ChatMessage("user", user))
