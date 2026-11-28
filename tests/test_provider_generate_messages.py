"""Provider multi-message capability and generate_chat delegation (#500 slice 3b-2)."""

from pathlib import Path

import pytest

from suzka.config import load_settings
from suzka.models.dummy_provider import DummyProvider
from suzka.models.transformers_provider import TransformersProvider
from suzka.persona import ConsciousAgent
from tests.test_model_providers import FakeChatProcessor, FakeModel


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _chat_provider(
    processor: FakeChatProcessor | None = None,
) -> tuple[TransformersProvider, FakeChatProcessor, FakeModel]:
    fake_model = FakeModel()
    processor = processor or FakeChatProcessor()
    provider = TransformersProvider(load_settings(CONFIG_PATH), model=fake_model, processor=processor)
    return provider, processor, fake_model


def test_generate_chat_golden_arguments_are_unchanged() -> None:
    provider, processor, model = _chat_provider()

    assert provider.generate_chat("SYS", "USR") == "4 5"

    # Literals captured from the pre-delegation implementation.
    assert processor.chat_calls == [
        (
            [
                {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
                {"role": "user", "content": [{"type": "text", "text": "USR"}]},
            ],
            {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "enable_thinking": False,
            },
        )
    ]
    assert model.generate_kwargs is not None
    assert sorted(model.generate_kwargs) == [
        "attention_mask",
        "do_sample",
        "input_ids",
        "max_new_tokens",
        "temperature",
        "top_p",
    ]
    assert model.generate_kwargs["max_new_tokens"] == 512
    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["temperature"] == 0.7
    assert model.generate_kwargs["top_p"] == 0.95
    assert model.generate_kwargs["input_ids"].tolist() == [[1, 2, 3]]


def test_generate_chat_still_accepts_empty_text() -> None:
    provider, processor, _ = _chat_provider()

    provider.generate_chat("", "")

    [(messages, _)] = processor.chat_calls
    assert messages[0]["content"] == [{"type": "text", "text": ""}]


def test_generate_messages_multi_turn_matches_expected_parts() -> None:
    provider, processor, model = _chat_provider()

    result = provider.generate_messages(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
    )

    assert result == "4 5"
    [(messages, kwargs)] = processor.chat_calls
    assert messages == [
        {"role": "system", "content": [{"type": "text", "text": "S"}]},
        {"role": "user", "content": [{"type": "text", "text": "u1"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": [{"type": "text", "text": "u2"}]},
    ]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["enable_thinking"] is False
    assert kwargs["tokenize"] is True
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["max_new_tokens"] == provider.settings.generation.max_new_tokens


def test_generate_messages_without_system_and_single_user() -> None:
    provider, processor, _ = _chat_provider()

    provider.generate_messages([{"role": "user", "content": "hi"}])

    assert processor.chat_calls[0][0] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


U = {"role": "user", "content": "u"}
A = {"role": "assistant", "content": "a"}
S = {"role": "system", "content": "S"}


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [S],
        [U, S, U],
        [S, S, U],
        [U, U],
        [A, U],
        [U, A],
        [S, U, A],
        [{"role": "user", "content": ""}],
        [{"role": "system", "content": ""}, U],
        [{"role": "user", "content": 5}],
        [{"role": "user"}],
        [{"role": "tool", "content": "x"}],
        [{"content": "x"}],
        ["user: hi"],
        "hello",
    ],
)
def test_generate_messages_rejects_invalid_shapes(messages) -> None:
    provider, processor, model = _chat_provider()

    with pytest.raises(ValueError):
        provider.generate_messages(messages)

    assert processor.chat_calls == []
    assert model.generate_kwargs is None


def test_generate_messages_requires_a_chat_template() -> None:
    processor = FakeChatProcessor()
    processor.chat_template = None
    provider, processor, _ = _chat_provider(processor)

    with pytest.raises(ValueError):
        provider.generate_messages([U])

    assert processor.chat_calls == []


class _Capable:
    """Provider stub with optional capabilities."""

    def __init__(self, roles: bool | None, has_messages: bool = True) -> None:
        self.calls: list = []
        if roles is not None:
            self.uses_chat_roles = lambda: roles
        if has_messages:
            self.generate_messages = self._record

    def _record(self, messages) -> str:
        self.calls.append(messages)
        return "ok"

    def generate(self, prompt: str) -> str:
        return "legacy"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (_Capable(None), True),
        (_Capable(True), True),
        (_Capable(False), False),
        (_Capable(True, has_messages=False), False),
        (_Capable(None, has_messages=False), False),
    ],
)
def test_conscious_agent_supports_messages_truth_table(provider, expected: bool) -> None:
    assert ConsciousAgent(provider).supports_messages() is expected


def test_conscious_agent_supports_messages_false_for_non_callable_attribute() -> None:
    provider = _Capable(True)
    provider.generate_messages = "nope"  # type: ignore[assignment]

    assert ConsciousAgent(provider).supports_messages() is False


def test_conscious_agent_supports_messages_false_for_dummy_provider() -> None:
    assert ConsciousAgent(DummyProvider()).supports_messages() is False


def test_conscious_agent_supports_messages_follows_template_of_transformers_provider() -> None:
    provider, _, _ = _chat_provider()
    assert ConsciousAgent(provider).supports_messages() is True
    plain = FakeChatProcessor()
    plain.chat_template = None
    provider, _, _ = _chat_provider(plain)
    assert ConsciousAgent(provider).supports_messages() is False


def test_conscious_agent_generate_messages_forwards_plain_messages() -> None:
    provider = _Capable(True)
    messages = [{"role": "user", "content": "hi"}]

    assert ConsciousAgent(provider).generate_messages(messages) == "ok"
    assert provider.calls == [messages]


def test_conscious_agent_generate_messages_refuses_without_capability() -> None:
    with pytest.raises(ValueError):
        ConsciousAgent(DummyProvider()).generate_messages([U])
    with pytest.raises(ValueError):
        ConsciousAgent(_Capable(False)).generate_messages([U])
