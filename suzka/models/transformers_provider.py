"""Hugging Face Transformers-backed model provider."""

from pathlib import Path
from typing import Any
import json

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from suzka.config import Settings


LOADABLE_ADAPTER_STATES = {"trial_active", "approved", "active"}
# Completion cue that ends PromptBuilder output; the chat template supplies its own turn start.
PLAIN_ASSISTANT_CUE = "\nAssistant:"
# Gemma 4 thought channel markers: <|channel>thought\n ... <channel|>answer
THOUGHT_CHANNEL_OPEN = "<|channel>"
THOUGHT_CHANNEL_CLOSE = "<channel|>"
PLAIN_THOUGHT_ECHO = "thought\n"


class TransformersProvider:
    """Provider backed by the configured Hugging Face Transformers model."""

    def __init__(
        self,
        settings: Settings,
        adapter_path: str | Path | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model_id = settings.model.primary_id
        self.processor = processor or AutoProcessor.from_pretrained(self.model_id)
        self.model = model or self._load_model(self.model_id)
        if adapter_path is not None:
            self.attach_adapter(adapter_path)

    def generate(self, prompt: str) -> str:
        inputs = self._generation_inputs(prompt)
        inputs = self._move_inputs_to_model_device(inputs)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.generation.max_new_tokens,
            "do_sample": self.settings.generation.do_sample,
        }
        if self.settings.generation.do_sample:
            generation_kwargs["temperature"] = self.settings.generation.temperature
            generation_kwargs["top_p"] = self.settings.generation.top_p
        output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        return self._decode_generated(generated_ids, chat=self._uses_chat_template())

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        if not target_text:
            raise ValueError("target_text must not be empty")

        self.model.eval()
        full_inputs = self._tokenize(context_text + target_text)
        context_inputs = self._tokenize(context_text) if context_text else None
        labels = full_inputs["input_ids"].clone()
        context_length = 0 if context_inputs is None else context_inputs["input_ids"].shape[1]
        labels[:, :context_length] = -100
        full_inputs = self._move_inputs_to_model_device(full_inputs)
        labels = labels.to(full_inputs["input_ids"].device)

        with torch.no_grad():
            outputs = self.model(**full_inputs, labels=labels)
        return float(outputs.loss.detach().cpu().item())

    def get_model(self) -> Any:
        return self.model

    def get_processor(self) -> Any:
        return self.processor

    def attach_adapter(self, adapter_path: str | Path) -> None:
        if not is_registry_approved_adapter(self.settings, adapter_path):
            raise ValueError("Adapter path is not approved by the adapter registry")
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))

    def _load_model(self, model_id: str) -> Any:
        load_kwargs: dict[str, Any] = {}
        if self.settings.model.device == "auto":
            load_kwargs["device_map"] = "auto"
        if self.settings.model.dtype != "auto":
            load_kwargs["torch_dtype"] = getattr(torch, self.settings.model.dtype)
        if self.settings.model.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        return AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    def _generation_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        """Tokenize a generation prompt, using the chat template when available.

        Instruction-tuned checkpoints ship a chat template and expect their
        turn markup; feeding them the plain completion prompt makes them echo
        prompt sections. Plain checkpoints keep the completion prompt as-is.
        """

        if not self._uses_chat_template():
            return self._tokenize(prompt)
        user_text = prompt.removesuffix(PLAIN_ASSISTANT_CUE).rstrip()
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}

    def _decode_generated(self, generated_ids: torch.Tensor, *, chat: bool = False) -> str:
        """Decode new tokens, mapping a model thought channel onto <think> tags.

        Gemma 4 can emit ``<|channel>thought\\n...<channel|>`` before the answer,
        even with thinking disabled. Skipping special tokens alone would leak
        the channel name and thought text, so the channel is rewritten into the
        <think> contract that ResponsePostprocessor keeps private. With the chat
        template the generation prompt already closes an empty thought channel,
        yet the model sometimes still writes the channel name as plain text;
        that leading "thought" line is dropped.
        """

        open_id = self._special_token_id(THOUGHT_CHANNEL_OPEN)
        close_id = self._special_token_id(THOUGHT_CHANNEL_CLOSE)
        ids = generated_ids.tolist()
        if open_id is None or close_id is None or open_id not in ids:
            text = self.processor.decode(generated_ids, skip_special_tokens=True)
            if chat and text.startswith(PLAIN_THOUGHT_ECHO):
                return text.removeprefix(PLAIN_THOUGHT_ECHO).lstrip()
            return text
        start = ids.index(open_id) + 1
        prefix = self.processor.decode(generated_ids[: start - 1], skip_special_tokens=True)
        if close_id not in ids[start:]:
            thought = self.processor.decode(generated_ids[start:], skip_special_tokens=True)
            return f"{prefix}<think>{_strip_channel_name(thought)}"
        end = ids.index(close_id, start)
        thought = self.processor.decode(generated_ids[start:end], skip_special_tokens=True)
        answer = self.processor.decode(generated_ids[end + 1 :], skip_special_tokens=True)
        return f"{prefix}<think>{_strip_channel_name(thought)}</think>{answer}"

    def _uses_chat_template(self) -> bool:
        return bool(getattr(self.processor, "chat_template", None))

    def _special_token_id(self, token: str) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or token not in getattr(tokenizer, "all_special_tokens", ()):
            return None
        token_id = tokenizer.convert_tokens_to_ids(token)
        return token_id if isinstance(token_id, int) else None

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.processor(text=text, return_tensors="pt")
        return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}

    def _move_inputs_to_model_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = getattr(self.model, "device", None)
        if device is None:
            return inputs
        return {key: value.to(device) for key, value in inputs.items()}


def is_registry_approved_adapter(settings: Settings, adapter_path: str | Path) -> bool:
    """Return whether an adapter path is registered in a loadable state."""

    registry_path = settings.adapter_registry.path
    if not registry_path.exists():
        return False

    requested_path = Path(adapter_path).expanduser().resolve()
    with registry_path.open("r", encoding="utf-8") as registry_file:
        registry_data = json.load(registry_file)
    for entry in _iter_registry_entries(registry_data):
        path = entry.get("path")
        state = entry.get("state")
        if path is None or state not in LOADABLE_ADAPTER_STATES:
            continue
        if Path(path).expanduser().resolve() == requested_path:
            return True
    return False


def _iter_registry_entries(registry_data: Any) -> list[dict[str, Any]]:
    if isinstance(registry_data, list):
        return [entry for entry in registry_data if isinstance(entry, dict)]
    if isinstance(registry_data, dict):
        adapters = registry_data.get("adapters")
        if isinstance(adapters, list):
            return [entry for entry in adapters if isinstance(entry, dict)]
        return [
            {"path": path, "state": state}
            for path, state in registry_data.items()
            if isinstance(path, str) and isinstance(state, str)
        ]
    return []


def _strip_channel_name(thought: str) -> str:
    return thought.removeprefix("thought").strip()
