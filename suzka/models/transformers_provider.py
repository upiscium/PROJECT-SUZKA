"""Hugging Face Transformers-backed model provider."""

from pathlib import Path
from collections.abc import Mapping, Sequence
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
        return self._generate_from(self._generation_inputs(prompt))

    def uses_chat_roles(self) -> bool:
        """Whether ``generate_chat`` can be used: only with a chat template."""

        return self._uses_chat_template()

    def generate_chat(self, system_text: str, user_text: str) -> str:
        """Generate with fixed instructions as the system message and the rest as the user message.

        Plain checkpoints have no roles; callers check ``uses_chat_roles``
        and send them the built completion prompt through ``generate``.
        """

        return self._generate_chat_parts(
            [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]
        )

    def generate_messages(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate a reply to a multi-message conversation.

        ``messages`` are plain ``{"role", "content"}`` mappings: an optional
        single ``system`` message first, then strictly alternating ``user`` /
        ``assistant`` messages, the last one ``user``. Each mapping must have
        exactly the keys ``role`` and ``content`` (unknown keys are rejected).
        Contents must be non-empty ``str``; user and assistant contents must
        also not be whitespace-only. Raises ``ValueError`` for any other shape
        or when the checkpoint has no chat template (callers check
        ``uses_chat_roles``). The input is validated once and only a
        normalised copy is used afterwards.

        Callers must only put code-generated text in a system message (#414:
        no user, model or stored text in the system role); the API cannot
        enforce this.
        """

        if not self._uses_chat_template():
            raise ValueError("this checkpoint has no chat template; use generate()")
        checked = _validate_messages(messages)
        return self._generate_chat_parts(
            [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in checked
            ]
        )

    def _generate_chat_parts(self, parts: list[dict[str, Any]]) -> str:
        # Shared by generate_chat (which never validated content) and generate_messages.
        if not self._uses_chat_template():
            raise ValueError("this checkpoint has no chat template; use generate()")
        return self._generate_from(self._chat_inputs(parts))

    def _generate_from(self, inputs: dict[str, torch.Tensor]) -> str:
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

    def release(self) -> None:
        """Drop the model and processor and give the GPU cache back (F15 DC-5:
        the supervisor releases a provider that served before it builds a new
        one, so two models are not alive during the new load).  The provider is
        unusable afterwards.  Not exercised on a real GPU in the tests."""

        self.model = None
        self.processor = None
        import gc

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
        return self._chat_inputs([{"role": "user", "content": [{"type": "text", "text": user_text}]}])

    def _chat_inputs(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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


def _validate_messages(messages: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    """Validate once and return a normalised copy; callers use only the copy."""

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise ValueError("messages must be a non-empty sequence")
    copy: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("each message must be a mapping with role and content")
        if set(message) != {"role", "content"}:
            raise ValueError("each message must have exactly the keys role and content")
        raw_role = message["role"]
        raw_content = message["content"]
        if not isinstance(raw_role, str) or str(raw_role) not in ("system", "user", "assistant"):
            raise ValueError("message role must be system, user or assistant")
        if not isinstance(raw_content, str) or not raw_content:
            raise ValueError("message content must be a non-empty string")
        role = str(raw_role)
        content = str(raw_content)
        if role != "system" and not content.strip():
            raise ValueError("user and assistant content must not be whitespace-only")
        copy.append({"role": role, "content": content})
    turns = copy[1:] if copy[0]["role"] == "system" else copy
    if not turns:
        raise ValueError("messages need at least one user message")
    for index, message in enumerate(turns):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("messages must alternate user/assistant after an optional leading system message")
    if turns[-1]["role"] != "user":
        raise ValueError("the last message must be from the user")
    return tuple(copy)


def _strip_channel_name(thought: str) -> str:
    return thought.removeprefix("thought").strip()
