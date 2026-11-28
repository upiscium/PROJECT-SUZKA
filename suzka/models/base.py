"""Shared model provider contracts."""

from typing import Any, Protocol


class ModelProvider(Protocol):
    """Protocol implemented by all model execution providers.

    Optional capabilities are discovered with ``getattr`` and are deliberately
    not part of this protocol: ``generate_chat(system_text, user_text)``,
    ``generate_messages(messages)`` (plain ``{"role", "content"}`` mappings:
    optional leading system, alternating user/assistant, last user) and
    ``uses_chat_roles()``. A provider without them (e.g. ``DummyProvider``)
    is served by ``generate`` only.
    """

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        """Calculate target-token loss conditioned on context text."""

    def get_model(self) -> Any:
        """Return the underlying model object."""

    def get_processor(self) -> Any:
        """Return the underlying processor/tokenizer object."""
