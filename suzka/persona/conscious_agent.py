"""Provider-agnostic conscious agent wrapper."""

import logging

from suzka.models import ModelProvider
from suzka.persona.prompt_builder import split_chat_prompt


logger = logging.getLogger(__name__)


class ConsciousAgent:
    """Generate raw model responses without postprocessing."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def generate(self, prompt: str, *, user_input: str | None = None) -> str:
        """Generate a reply; with ``user_input``, use separate chat roles if the provider can.

        Only the fixed instructions go in the system role; the runtime data
        and the user's message go in the user role. Providers that cannot use
        roles (no ``generate_chat``, or ``uses_chat_roles()`` false, e.g. a
        checkpoint without a chat template) get the built prompt unchanged.
        """

        generate_chat = getattr(self.provider, "generate_chat", None)
        uses_roles = getattr(self.provider, "uses_chat_roles", None)
        if (
            user_input is not None
            and callable(generate_chat)
            and (not callable(uses_roles) or uses_roles())
        ):
            split = split_chat_prompt(prompt, user_input)
            if split is not None:
                system_text, user_text = split
                return str(generate_chat(system_text, user_text))
            logger.debug("prompt has no role boundary; sending it as one message")
        return self.provider.generate(prompt)
