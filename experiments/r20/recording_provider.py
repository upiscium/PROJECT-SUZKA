"""Lab-only RECORDING wrapper of a model provider (smoke diagnostics; observation only).

``RecordingProvider`` delegates everything to the provider it wraps and appends, for every
``generate`` / ``generate_chat`` call, ONE row to a 0600 jsonl in the subject's private directory:
the exact prompt (or system / user text) as passed to the inner provider, the sampling parameters
the inner provider will use (read from ``inner.settings`` when it has them, else ``null``) and the
output.  It never changes the text, never retries, never judges.

It exists so a Human can tell apart (1) sampling / quantisation noise, (2) how the production chat
path assembles the model input, and (3) the model itself.  It is OPT-IN: ``serve --record-calls``
(set by the smoke only when asked).  Without that flag nothing here is imported or installed.

Raw prompts and outputs live ONLY in the private file.  ``assembly_numbers`` turns a row into
numbers and booleans (no text) for the results file.

Nothing under ``suzka/`` is touched: ``serve`` wraps an injected provider, or replaces the loader
name the app's lifespan calls (so every rebuilt cycle gets a wrapper again).
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

from suzka.runtime.conversation_trace import RECENT_CONVERSATION_HEADING

CALLS_NAME = "model_calls.jsonl"  # in the subject's private directory (0700 dir, 0600 file)
_EPISODIC_HEADING = "Stored episodic evidence:"
_SEMANTIC_HEADING = "Stored semantic evidence:"
_TRACE_LINE = re.compile(r'^- .+ — User: ".*" / You: ', re.MULTILINE)
_PREFIX_CHARS = 20
_WM_EPISODIC_LINE = re.compile(r'^\{"source_kind":"episodic"', re.MULTILINE)


def _present(needle: str, text: str) -> bool:
    return needle in text or json.dumps(needle)[1:-1] in text


def _sampling_of(inner: Any) -> dict[str, Any] | None:
    """The generation parameters ``inner`` will use, or None when it exposes none (e.g. a fixture)."""

    settings = getattr(inner, "settings", None)
    generation = getattr(settings, "generation", None)
    if generation is None:
        return None
    try:
        do_sample = bool(generation.do_sample)
        found: dict[str, Any] = {
            "max_new_tokens": int(generation.max_new_tokens),
            "do_sample": do_sample,
            # the provider passes temperature / top_p to generate() only when sampling
            "temperature": float(generation.temperature) if do_sample else None,
            "top_p": float(generation.top_p) if do_sample else None,
        }
    except (AttributeError, TypeError, ValueError):
        return None
    model = getattr(settings, "model", None)
    found["model_id"] = getattr(model, "primary_id", None)
    found["load_in_4bit"] = getattr(model, "load_in_4bit", None)
    return found


def _append(path: Path, row: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class RecordingProvider:
    """Delegates everything to ``inner``; ``generate`` / ``generate_chat`` also record one row.

    ``generate_chat`` exists on the wrapper only when the inner provider has it (the app probes it).
    """

    def __init__(self, inner: Any, private_dir: Path) -> None:
        self._inner = inner
        self._path = Path(private_dir) / CALLS_NAME
        self._lock = threading.Lock()
        self._sequence = 0

    def __getattr__(self, name: str) -> Any:  # only called for names not found on the wrapper
        if name in {"_inner", "_path", "_lock", "_sequence"}:
            raise AttributeError(name)
        target = getattr(self._inner, name)  # AttributeError when the inner has none: callers probe with getattr
        return self._generate_chat if name == "generate_chat" else target

    def _record(self, row: dict[str, Any], call: Callable[[], Any]) -> str:
        sampling = _sampling_of(self._inner)  # read per call: the inner may be reconfigured
        try:
            output = str(call())
        except BaseException as error:
            self._write({**row, **self._extra(), "sampling": sampling, "output": None, "error": type(error).__name__})
            raise
        self._write({**row, **self._extra(), "sampling": sampling, "output": output})
        return output

    def _extra(self) -> dict[str, Any]:
        """The report of a lab adapter inside this wrapper (`chat_role_adapter`), if any, as `{"chat_roles": ...}`."""

        take = getattr(self._inner, "take_record_extra", None)
        report = take() if callable(take) else None
        return {} if report is None else {"chat_roles": report}

    def _write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            _append(self._path, {"seq": self._sequence, **row})

    def generate(self, prompt: str) -> str:
        return self._record({"method": "generate", "prompt": prompt}, lambda: self._inner.generate(prompt))

    def _generate_chat(self, system_text: str, user_text: str) -> str:
        return self._record(
            {"method": "generate_chat", "system_text": system_text, "user_text": user_text},
            lambda: self._inner.generate_chat(system_text, user_text),
        )


def wrap_loader(loader: Callable[..., Any], private_dir: Path) -> Callable[..., Any]:
    """A drop-in for ``load_model_provider``: every provider it builds is wrapped."""

    def load(*arguments: Any, **keywords: Any) -> RecordingProvider:
        return RecordingProvider(loader(*arguments, **keywords), private_dir)

    return load


def read_calls(private_dir: Path) -> list[dict[str, Any]]:
    """Every recorded row so far (``[]`` when recording was off).  Contains RAW text: keep it private."""

    path = Path(private_dir) / CALLS_NAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _count_block(text: str, heading: str) -> int:
    """Evidence lines under ``heading`` (up to the next blank line), without the ``- none`` placeholder."""

    index = text.find(heading)
    if index < 0:
        return 0
    block = text[index + len(heading):].split("\n\n", 1)[0].splitlines()
    return sum(1 for line in block if line.startswith("- ") and line.strip() != "- none")


def model_input_text(row: dict[str, Any]) -> str:
    if row.get("method") == "generate_chat":
        return f"{row.get('system_text', '')}\n{row.get('user_text', '')}"
    return str(row.get("prompt", ""))


def assembly_numbers(
    row: dict[str, Any], *, previous_request: str | None = None, previous_response: str | None = None
) -> dict[str, Any]:
    """Numbers and booleans about ONE call's model input (NO text), labelled ``observed``.

    ``messages``: how many messages the chat path hands the model (``generate_chat``: 1 system + 1 user;
    ``generate``: the prompt, which a chat-template provider wraps as one user message).
    ``prior_turn_messages`` is therefore 0 for both.  Earlier turns can only enter as TEXT inside the
    user part: ``trace_turns`` (ConversationTrace lines) and ``episodic_items`` / ``semantic_items``
    (retrieved memory evidence lines).  ``working_memory_episodic_items`` counts the episodic Working Memory evidence lines (earlier turns can
    also arrive there, JSON-escaped; the trace skips turns Working Memory already shows).
    ``previous_*_in_input`` are substring checks (plain or JSON-escaped) of the immediately
    preceding turn's request / the first 20 characters of its reply against the model input.
    """

    chat = row.get("method") == "generate_chat"
    system = str(row.get("system_text", "")) if chat else ""
    user = str(row.get("user_text", "")) if chat else str(row.get("prompt", ""))
    text = f"{system}\n{user}" if chat else user
    found: dict[str, Any] = {
        "kind": "observed",
        "method": row.get("method"),
        "messages": 2 if chat else 1,
        "prior_turn_messages": 0,
        "system_chars": len(system),
        "user_chars": len(user),
        "prompt_chars": len(system) + len(user),
        "trace_block_present": RECENT_CONVERSATION_HEADING in text,
        "trace_turns": len(_TRACE_LINE.findall(text)),
        "episodic_items": _count_block(text, _EPISODIC_HEADING),
        "semantic_items": _count_block(text, _SEMANTIC_HEADING),
        # Working Memory evidence lines are JSON (non-ASCII escaped), so a turn can be in the input
        # either as plain text or as that escaped text; both count as present.
        "working_memory_episodic_items": len(_WM_EPISODIC_LINE.findall(text)),
        "previous_request_in_input": None if previous_request is None else _present(previous_request, text),
        "previous_reply_in_input": None
        if not previous_response
        else _present(previous_response[:_PREFIX_CHARS], text),
    }
    return found


__all__ = [
    "CALLS_NAME", "RecordingProvider", "assembly_numbers", "model_input_text", "read_calls", "wrap_loader",
]
