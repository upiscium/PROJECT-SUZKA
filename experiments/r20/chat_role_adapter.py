"""Lab-only CHAT-ROLE adapter of a model provider (smoke diagnostics; observation only).

Hypothesis under test: the production chat path hands the model exactly two messages, a fixed
system header and ONE user message that holds everything, including the rendered ``Recent
conversation`` block.  A 12B model may use the earlier turns better when they arrive as real
alternating chat roles.  ``ChatRoleAdapter`` rewrites ONLY the message structure of
``generate_chat``:

    [system, (user, assistant) * N oldest -> newest, final user]

where the pairs are parsed out of the rendered block, and the final user message is the original
user text with that block removed (``User message: ...`` is kept).  Nothing else changes: the same
inner provider, the same generation settings (they live in the inner provider), no retry, no
normalisation.

It is OPT-IN (``serve --chat-roles``, set by the smoke only when asked) and refuses on any doubt:
on a missing / duplicated block, any line that is not exactly the renderer's turn format, an inner
provider without the private chat path, ... it passes the call through UNCHANGED and reports
``adapter_skipped: <reason label>``.  It never guesses.

LAB-ONLY use of PRIVATE names of ``suzka.models.transformers_provider.TransformersProvider``
(``_uses_chat_template``, ``_chat_inputs``, ``_generate_from``): ``generate_chat`` only accepts
``(system_text, user_text)``, so several messages need the lower-level path.  Nothing under
``suzka/`` is touched or imported here beyond two public constants.

Composition: the adapter sits INSIDE the ``RecordingProvider`` (recorder -> adapter -> model), so the
recorder sees the original two-message form (kept for comparison) and, through ``take_record_extra``,
the adapter's report incl. the messages actually sent.  Raw text goes only to the private 0600 file.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import re
import threading
from typing import Any

from suzka.persona.prompt_builder import CURRENT_MESSAGE_HEADING
from suzka.runtime.conversation_trace import RECENT_CONVERSATION_HEADING

_QUOTED = r'"(?:[^"\\]|\\.)*"'  # a json.dumps string: no raw quote, no raw newline
_TURN_LINE = re.compile(rf"- (?P<label>.+?) — User: (?P<user>{_QUOTED}) / You: (?P<reply>{_QUOTED})")
_FINAL_MARK = f"\n\n{CURRENT_MESSAGE_HEADING}\n"
_PRIVATE_PATH = ("_uses_chat_template", "_chat_inputs", "_generate_from")

# adapter_skipped labels
NO_BLOCK = "no_trace_block"
DUPLICATE_BLOCK = "duplicate_trace_block"
NO_FINAL_MARK = "no_user_message_marker"
BLOCK_OUTSIDE_RUNTIME_DATA = "trace_block_after_user_message"
BLOCK_NOT_LINE_START = "trace_heading_not_at_line_start"
EMPTY_BLOCK = "empty_trace_block"
UNPARSEABLE_LINE = "unparseable_trace_line"
NO_PRIVATE_PATH = "inner_lacks_private_path"
NO_CHAT_TEMPLATE = "inner_has_no_chat_template"
NOT_A_CHAT_CALL = "not_a_chat_call"  # the app used `generate(prompt)`: there is no system / user split to restructure


class ParseRefused(Exception):
    """The rendered block cannot be parsed exactly: carry the reason label (never text)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def split_trace(user_text: str) -> tuple[list[tuple[str, str]], str]:
    """``(pairs oldest first, user text without the block)``; raises ``ParseRefused`` on any doubt."""

    head, mark, tail = user_text.partition(_FINAL_MARK)  # the builder's own marker is the FIRST one
    if not mark:
        raise ParseRefused(NO_FINAL_MARK)
    count = head.count(RECENT_CONVERSATION_HEADING)
    if count == 0:
        if RECENT_CONVERSATION_HEADING in tail:
            raise ParseRefused(BLOCK_OUTSIDE_RUNTIME_DATA)
        raise ParseRefused(NO_BLOCK)
    if count > 1:
        raise ParseRefused(DUPLICATE_BLOCK)
    start = head.index(RECENT_CONVERSATION_HEADING)
    if start and head[start - 1] != "\n":
        raise ParseRefused(BLOCK_NOT_LINE_START)
    end = head.find("\n\n", start)
    block_end = len(head) if end < 0 else end
    lines = head[start:block_end].split("\n")[1:]  # drop the heading line
    if not lines:
        raise ParseRefused(EMPTY_BLOCK)
    pairs: list[tuple[str, str]] = []
    for line in lines:
        found = _TURN_LINE.fullmatch(line)
        if found is None:  # includes the command-turn line: not a chat pair, so not guessed at
            raise ParseRefused(UNPARSEABLE_LINE)
        try:
            user, reply = json.loads(found["user"]), json.loads(found["reply"])
        except ValueError as error:
            raise ParseRefused(UNPARSEABLE_LINE) from error
        if not isinstance(user, str) or not isinstance(reply, str):
            raise ParseRefused(UNPARSEABLE_LINE)
        pairs.append((user, reply))
    before = head[:start]
    after = "" if end < 0 else head[end + 2 :]
    if not after and before.endswith("\n\n"):
        before = before[:-2]
    return pairs, f"{before}{after}{mark}{tail}"


def build_messages(system_text: str, pairs: list[tuple[str, str]], final_user: str) -> list[dict[str, Any]]:
    """The structure sent to the model: system, (user, assistant) pairs oldest first, final user."""

    def message(role: str, text: str) -> dict[str, Any]:
        return {"role": role, "content": [{"type": "text", "text": text}]}

    messages = [message("system", system_text)]
    for user, reply in pairs:
        messages += [message("user", user), message("assistant", reply)]
    messages.append(message("user", final_user))
    return messages


def _has_private_path(inner: Any) -> bool:
    return all(callable(getattr(inner, name, None)) for name in _PRIVATE_PATH)


class ChatRoleAdapter:
    """Delegates everything to ``inner``; ``generate_chat`` sends real chat roles when it can.

    ``generate_chat`` exists on the wrapper only when the inner provider has it (the app probes it).
    ``take_record_extra`` hands the report of the LAST call in this thread to the recorder.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._local = threading.local()

    def __getattr__(self, name: str) -> Any:  # only called for names not found on the wrapper
        if name in {"_inner", "_local"}:
            raise AttributeError(name)
        target = getattr(self._inner, name)
        return self._generate_chat if name == "generate_chat" else target

    def take_record_extra(self) -> dict[str, Any] | None:
        report = getattr(self._local, "report", None)
        self._local.report = None
        return report

    def generate(self, prompt: str) -> str:
        self._local.report = {
            "kind": "observed", "applied": False, "adapter_skipped": NOT_A_CHAT_CALL, "pairs_moved": 0,
            "user_chars_before": len(prompt), "user_chars_after": len(prompt),
        }
        return self._inner.generate(prompt)

    def _generate_chat(self, system_text: str, user_text: str) -> str:
        report: dict[str, Any] = {
            "kind": "observed", "applied": False, "adapter_skipped": None, "pairs_moved": 0,
            "user_chars_before": len(user_text), "user_chars_after": len(user_text),
        }
        self._local.report = report
        inner = self._inner
        reason: str | None = None
        pairs: list[tuple[str, str]] = []
        final_user = user_text
        if not _has_private_path(inner):
            reason = NO_PRIVATE_PATH
        elif not inner._uses_chat_template():
            reason = NO_CHAT_TEMPLATE
        else:
            try:
                pairs, final_user = split_trace(user_text)
            except ParseRefused as refusal:
                reason = refusal.reason
        if reason is not None:
            report["adapter_skipped"] = reason
            return inner.generate_chat(system_text, user_text)  # unchanged, exactly as without the adapter
        messages = build_messages(system_text, pairs, final_user)
        report.update(
            applied=True, pairs_moved=len(pairs), user_chars_after=len(final_user),
            messages_sent=messages,  # RAW text: private file only
        )
        return inner._generate_from(inner._chat_inputs(messages))


def wrap_loader(loader: Callable[..., Any]) -> Callable[..., Any]:
    """A drop-in for ``load_model_provider``: every provider it builds is wrapped."""

    def load(*arguments: Any, **keywords: Any) -> ChatRoleAdapter:
        return ChatRoleAdapter(loader(*arguments, **keywords))

    return load


def numbers(row: dict[str, Any]) -> dict[str, Any] | None:
    """The adapter's report of a recorded call WITHOUT the messages (numbers and labels only)."""

    report = row.get("chat_roles")
    if not isinstance(report, dict):
        return None
    return {key: value for key, value in report.items() if key != "messages_sent"}


__all__ = ["ChatRoleAdapter", "ParseRefused", "build_messages", "numbers", "split_trace", "wrap_loader"]
