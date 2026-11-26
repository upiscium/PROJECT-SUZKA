"""Shared helpers of the launch / model-config tests (no test here).  Nothing here starts a real
model: the terminal, the GPU probe and the HF cache are fakes."""

from __future__ import annotations

from pathlib import Path

from experiments.r20.guards import GuardRefusal
from experiments.r20.launch import HumanLaunch, _attest, confirmation_sentence
from experiments.r20.model_config import ModelOverride, cache_dir_name

REVISION = "0123456789abcdef0123456789abcdef01234567"
MODEL_ID = "example-org/example-12b-it"


class FakeTerminal:
    """A terminal that types ``answer`` (or the exact sentence)."""

    def __init__(self, *, tty: bool = True, foreground: bool = True, answer: str | None = None):
        self.tty = tty
        self.foreground = foreground
        self.answer = answer
        self.prompts: list[str] = []

    def is_tty(self) -> bool:
        return self.tty

    def is_foreground(self) -> bool:
        return self.foreground

    def prompt(self, text: str) -> str:
        self.prompts.append(text)
        if self.answer is not None:
            return self.answer
        # the Human reads the sentence off the screen and types it
        for line in text.splitlines():
            if line.strip().startswith("I AM THE HUMAN OPERATOR"):
                return line.strip()
        raise AssertionError("no sentence was displayed")


def make_cache(hf_home: Path, *, revision: str = REVISION, model_id: str = MODEL_ID, snapshot: bool = True) -> Path:
    root = hf_home / "hub" / cache_dir_name(model_id)
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    if snapshot:
        (root / "snapshots" / revision).mkdir(parents=True)
    return hf_home


def real_model(hf_home: Path | None = None, **changes: object) -> ModelOverride:
    values: dict[str, object] = {
        "provider": "transformers", "model_id": MODEL_ID, "revision": REVISION, "hf_home": hf_home or Path("/models/hf"),
        "max_new_tokens": 256, "load_in_4bit": True,
    }
    values.update(changes)
    return ModelOverride(**values)  # type: ignore[arg-type]


def human_launch(stage: str = "S0", digest: str = "a" * 64) -> HumanLaunch:
    """A launch minted the only legal way: through the guard, with a fake terminal."""

    return _attest(stage, digest, human_flag=True, terminal=FakeTerminal(), environ={}, ancestors=lambda: [])


def refusal_code(function: object, *args: object, **kwargs: object) -> str:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except GuardRefusal as refusal:
        return refusal.code
    raise AssertionError("no refusal")


def sentence(stage: str, digest: str, revision8: str = "") -> str:
    return confirmation_sentence(stage, digest, revision8)
