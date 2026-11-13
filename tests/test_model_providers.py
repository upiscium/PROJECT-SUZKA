from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from suzka.config import load_settings
from suzka.models.dummy_provider import DummyProvider
from suzka.models.model_loader import load_model_provider
from suzka.models.transformers_provider import (
    TransformersProvider,
    is_registry_approved_adapter,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class FakeProcessor:
    def __call__(self, *, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        token_ids = list(range(1, len(text.split()) + 1)) or [0]
        return {"input_ids": torch.tensor([token_ids])}

    def decode(self, output_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        return " ".join(str(token_id) for token_id in output_ids.tolist())


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.labels_seen: torch.Tensor | None = None
        self.generate_kwargs: dict | None = None

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs) -> SimpleNamespace:
        self.labels_seen = kwargs["labels"]
        return SimpleNamespace(loss=torch.tensor(2.5))

    def generate(self, **kwargs) -> torch.Tensor:
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_dummy_provider_is_deterministic() -> None:
    provider = DummyProvider()

    assert provider.generate("hello") == provider.response_text
    assert provider.calculate_loss("context", "target") == provider.loss_value


def test_model_loader_defaults_to_dummy_provider() -> None:
    provider = load_model_provider(load_settings(CONFIG_PATH))

    assert isinstance(provider, DummyProvider)


def test_transformers_provider_rejects_empty_target_text() -> None:
    provider = TransformersProvider(
        load_settings(CONFIG_PATH),
        model=FakeModel(),
        processor=FakeProcessor(),
    )

    with pytest.raises(ValueError, match="target_text"):
        provider.calculate_loss("context", "")


def test_transformers_loss_masks_context_tokens() -> None:
    fake_model = FakeModel()
    provider = TransformersProvider(
        load_settings(CONFIG_PATH),
        model=fake_model,
        processor=FakeProcessor(),
    )

    loss = provider.calculate_loss("one two", " three four")

    assert loss == 2.5
    assert fake_model.eval_called
    assert fake_model.labels_seen is not None
    assert fake_model.labels_seen.tolist() == [[-100, -100, 3, 4]]


def test_transformers_provider_loads_configured_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: dict[str, object] = {}

    def fake_processor_from_pretrained(model_id: str) -> FakeProcessor:
        loaded["processor_model_id"] = model_id
        return FakeProcessor()

    def fake_model_from_pretrained(model_id: str, **kwargs) -> FakeModel:
        loaded["model_model_id"] = model_id
        loaded["model_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(
        "suzka.models.transformers_provider.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "suzka.models.transformers_provider.AutoModelForImageTextToText.from_pretrained",
        fake_model_from_pretrained,
    )

    settings = load_settings(CONFIG_PATH)
    TransformersProvider(settings)

    assert loaded["processor_model_id"] == settings.model.primary_id
    assert loaded["model_model_id"] == settings.model.primary_id
    assert "quantization_config" in loaded["model_kwargs"]


def test_transformers_generate_decodes_only_new_tokens() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(settings, model=fake_model, processor=FakeProcessor())

    generated = provider.generate("prompt has three")

    assert generated == "4 5"


class FakeChatProcessor(FakeProcessor):
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.chat_calls: list[tuple[list[dict], dict]] = []
        self.plain_calls: list[str] = []

    def __call__(self, *, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        self.plain_calls.append(text)
        return super().__call__(text=text, return_tensors=return_tensors)

    def apply_chat_template(self, messages: list[dict], **kwargs) -> dict[str, torch.Tensor]:
        self.chat_calls.append((messages, kwargs))
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}


def test_transformers_generate_uses_chat_template_for_instruction_models() -> None:
    fake_model = FakeModel()
    processor = FakeChatProcessor()
    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(settings, model=fake_model, processor=processor)

    generated = provider.generate("Context: private\n\nUser: hello\nAssistant:")

    assert generated == "4 5"
    assert processor.plain_calls == []
    [(messages, kwargs)] = processor.chat_calls
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "Context: private\n\nUser: hello"}]}
    ]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["enable_thinking"] is False
    assert fake_model.generate_kwargs is not None
    assert "attention_mask" in fake_model.generate_kwargs


def test_transformers_loss_keeps_plain_text_with_chat_template() -> None:
    fake_model = FakeModel()
    processor = FakeChatProcessor()
    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(settings, model=fake_model, processor=processor)

    provider.calculate_loss("context words", " target")

    assert processor.chat_calls == []
    assert processor.plain_calls == ["context words target", "context words"]


def test_transformers_generate_omits_sampling_kwargs_when_not_sampling() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={"generation": settings.generation.model_copy(update={"do_sample": False})}
    )
    provider = TransformersProvider(settings, model=fake_model, processor=FakeProcessor())

    provider.generate("hello")

    assert fake_model.generate_kwargs is not None
    assert "temperature" not in fake_model.generate_kwargs
    assert "top_p" not in fake_model.generate_kwargs
    assert fake_model.generate_kwargs["do_sample"] is False


def test_transformers_generate_includes_sampling_kwargs_when_sampling() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={"generation": settings.generation.model_copy(update={"do_sample": True})}
    )
    provider = TransformersProvider(settings, model=fake_model, processor=FakeProcessor())

    provider.generate("hello")

    assert fake_model.generate_kwargs is not None
    assert fake_model.generate_kwargs["temperature"] == settings.generation.temperature
    assert fake_model.generate_kwargs["top_p"] == settings.generation.top_p
    assert fake_model.generate_kwargs["do_sample"] is True


def test_adapter_paths_must_be_approved_by_registry(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"adapters": [{"path": str(adapter_path), "state": "approved"}]}),
        encoding="utf-8",
    )
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": registry_path}
            )
        }
    )

    assert is_registry_approved_adapter(settings, adapter_path)
    assert not is_registry_approved_adapter(settings, tmp_path / "unregistered")


class FakeChannelTokenizer:
    all_special_tokens = ["<|channel>", "<channel|>"]
    ids = {"<|channel>": 100, "<channel|>": 101}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids[token]


class FakeChannelProcessor(FakeProcessor):
    tokenizer = FakeChannelTokenizer()
    vocab = {100: "<|channel>", 101: "<channel|>", 7: "thought", 8: "pondering", 9: "hello"}

    def decode(self, output_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        words = [self.vocab[token_id] for token_id in output_ids.tolist()]
        if skip_special_tokens:
            words = [word for word in words if word not in FakeChannelTokenizer.ids]
        return " ".join(words)


class FakeChannelModel(FakeModel):
    def __init__(self, new_tokens: list[int]) -> None:
        super().__init__()
        self.new_tokens = new_tokens

    def generate(self, **kwargs) -> torch.Tensor:
        self.generate_kwargs = kwargs
        return torch.tensor([kwargs["input_ids"][0].tolist() + self.new_tokens])


@pytest.mark.parametrize(
    ("new_tokens", "expected"),
    [
        ([100, 7, 101, 9], "<think></think>hello"),
        ([100, 7, 8, 101, 9], "<think>pondering</think>hello"),
        ([100, 7, 8], "<think>pondering"),
        ([9], "hello"),
    ],
)
def test_transformers_generate_maps_thought_channel_to_think_tags(
    new_tokens: list[int], expected: str
) -> None:
    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(
        settings, model=FakeChannelModel(new_tokens), processor=FakeChannelProcessor()
    )

    assert provider.generate("hi") == expected


def test_thought_channel_never_reaches_visible_response() -> None:
    from suzka.persona.response_postprocessor import ResponsePostprocessor

    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(
        settings, model=FakeChannelModel([100, 7, 8, 101, 9]), processor=FakeChannelProcessor()
    )

    processed = ResponsePostprocessor().process(provider.generate("hi"))

    assert processed.visible_response == "hello"
    assert processed.hidden_thought == "pondering"


class FakeEchoProcessor(FakeChatProcessor):
    def decode(self, output_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        return "thought\nhello there"


def test_chat_generation_drops_plain_thought_channel_echo() -> None:
    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(settings, model=FakeModel(), processor=FakeEchoProcessor())

    assert provider.generate("hi") == "hello there"


def test_plain_generation_keeps_leading_thought_text() -> None:
    class PlainEchoProcessor(FakeProcessor):
        def decode(self, output_ids: torch.Tensor, skip_special_tokens: bool) -> str:
            return "thought\nhello there"

    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(settings, model=FakeModel(), processor=PlainEchoProcessor())

    assert provider.generate("hi") == "thought\nhello there"
