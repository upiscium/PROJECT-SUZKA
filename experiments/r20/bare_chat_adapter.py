"""Lab-only BARE CHAT adapter of a model provider (smoke control; observation only).

Question under test: when the 12B model misreads a follow-up ('rephrase my last answer'), is it the
model (or its 4-bit quantisation), or SUZKA's large structured prompt (values, goals, commitments,
emotion, working-memory evidence ...) that steers it?  The decisive control is a plain chat:

    [system: BARE_SYSTEM_TEXT, (user, assistant) * N oldest -> newest, current user message]

``BareChatAdapter`` rewrites the model INPUT of ``generate_chat`` only.  The current user message is
the text after the builder's ``User message:`` marker; the history is the adapter's OWN list of
(current user message, model output) pairs of its earlier calls, in order.  Nothing of SUZKA's prompt
(header, runtime data, trace block, evidence) is sent.  The SUZKA runtime around it runs normally
(memory writes, ...); only what the model sees is bare.  Same inner provider, same generation settings
(they live in the inner provider), no retry, no normalisation.

OPT-IN (``serve --bare-chat``, set by the smoke only when asked).  On any doubt (no private path, no
chat template, the marker is not present exactly once, an empty message, a ``generate(prompt)`` call)
the call is passed through UNCHANGED and reported as ``adapter_skipped: <label>``; such a call adds
nothing to the history.  The history lives in this object: one provider object per serve process and
one process per subject, so it starts empty for every subject.

LAB-ONLY use of the PRIVATE names of ``TransformersProvider`` (``_uses_chat_template``,
``_chat_inputs``, ``_generate_from``), exactly as ``chat_role_adapter`` does.  Nothing under
``suzka/`` is touched.  Composition: recorder -> adapter -> model; the recorder keeps the original
two-message form and, through ``take_record_extra``, the report under the key ``bare_chat``.  Raw
text (``messages_sent``) goes only to the private 0600 file.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

from experiments.r20.chat_role_adapter import (
    NO_CHAT_TEMPLATE, NO_PRIVATE_PATH, NOT_A_CHAT_CALL, _has_private_path, build_messages,
)
from suzka.persona.prompt_builder import CURRENT_MESSAGE_HEADING

BARE_SYSTEM_TEXT = "あなたは親切なAIアシスタントです。日本語で簡潔に答えてください。"
RECORD_KEY = "bare_chat"
_MARK = f"\n\n{CURRENT_MESSAGE_HEADING}\n"

NO_USER_MESSAGE = "no_user_message"


def extract_user_message(user_text: str) -> str | None:
    """The current user message: the text after the builder's marker; ``None`` unless the marker occurs exactly once.

    The builder renders ``<runtime data>\\n\\nUser message:\\n<user input>``.  A second marker (a message or
    a stored line imitating it) makes the split ambiguous, so it is refused rather than guessed.
    """

    if user_text.count(_MARK) != 1:
        return None
    message = user_text.partition(_MARK)[2]
    return message if message.strip() else None


class BareChatAdapter:
    """Delegates everything to ``inner``; ``generate_chat`` sends a bare chat when it can.

    ``generate_chat`` exists on the wrapper only when the inner provider has it (the app probes it).
    ``take_record_extra`` hands the report of the LAST call in this thread to the recorder.
    """

    record_extra_key = RECORD_KEY

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._local = threading.local()
        self._history: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:  # only called for names not found on the wrapper
        if name in {"_inner", "_local", "_history", "_lock"}:
            raise AttributeError(name)
        target = getattr(self._inner, name)
        return self._generate_chat if name == "generate_chat" else target

    def take_record_extra(self) -> dict[str, Any] | None:
        report = getattr(self._local, "report", None)
        self._local.report = None
        return report

    @staticmethod
    def _report(chars: int, reason: str | None) -> dict[str, Any]:
        return {
            "kind": "observed", "applied": False, "adapter_skipped": reason, "messages_sent_count": 0,
            "user_chars_before": chars, "user_chars_after": chars, "history_pairs": 0,
        }

    def generate(self, prompt: str) -> str:
        self._local.report = self._report(len(prompt), NOT_A_CHAT_CALL)
        return self._inner.generate(prompt)

    def _generate_chat(self, system_text: str, user_text: str) -> str:
        report = self._report(len(user_text), None)
        self._local.report = report
        inner = self._inner
        reason: str | None = None
        message: str | None = None
        if not _has_private_path(inner):
            reason = NO_PRIVATE_PATH
        elif not inner._uses_chat_template():
            reason = NO_CHAT_TEMPLATE
        else:
            message = extract_user_message(user_text)
            if message is None:
                reason = NO_USER_MESSAGE
        if reason is not None or message is None:
            report["adapter_skipped"] = reason
            return inner.generate_chat(system_text, user_text)  # unchanged, exactly as without the adapter
        with self._lock:
            history = list(self._history)
        messages = build_messages(BARE_SYSTEM_TEXT, history, message)
        report.update(
            applied=True, messages_sent_count=len(messages), user_chars_after=len(message),
            history_pairs=len(history),
            messages_sent=messages,  # RAW text: private file only
        )
        output = inner._generate_from(inner._chat_inputs(messages))
        with self._lock:
            self._history.append((message, str(output)))
        return output


def wrap_loader(loader: Callable[..., Any]) -> Callable[..., Any]:
    """A drop-in for ``load_model_provider``: every provider it builds is wrapped (one history per provider)."""

    def load(*arguments: Any, **keywords: Any) -> BareChatAdapter:
        return BareChatAdapter(loader(*arguments, **keywords))

    return load


def numbers(row: dict[str, Any]) -> dict[str, Any] | None:
    """The adapter's report of a recorded call WITHOUT the messages (numbers and labels only)."""

    report = row.get(RECORD_KEY)
    if not isinstance(report, dict):
        return None
    return {key: value for key, value in report.items() if key != "messages_sent"}


__all__ = ["BARE_SYSTEM_TEXT", "BareChatAdapter", "extract_user_message", "numbers", "wrap_loader"]
