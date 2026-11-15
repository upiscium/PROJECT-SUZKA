"""Provider-agnostic conscious agent wrapper."""

from suzka.models import ModelProvider
from suzka.persona.prompt_builder import split_chat_prompt


class ConsciousAgent:
    """Generate raw model responses without postprocessing."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def generate(self, prompt: str, *, user_input: str | None = None) -> str:
        """Generate a reply; with ``user_input``, use separate chat roles if the provider can.

        Instructions and private runtime data go in the system role and only
        the user's own text in the user role. Providers without
        ``generate_chat`` keep the single completion prompt.
        """

        generate_chat = getattr(self.provider, "generate_chat", None)
        if user_input is not None and callable(generate_chat):
            split = split_chat_prompt(prompt, user_input)
            if split is not None:
                system_text, user_text = split
                return str(generate_chat(system_text, user_text))
        return self.provider.generate(prompt)
