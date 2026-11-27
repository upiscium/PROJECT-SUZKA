"""Pure chat model-input composition (design Issue #500, slice 2).

This module assembles the list of chat messages sent to a chat-template model from already
prepared text. It performs no I/O and reads no database, ``AgentState`` or clock. Nothing imports
it yet; it is wired into the main loop behind a setting in a later slice.

Modes (names are DEFAULTS, not yet fixed by the Human):

``legacy``
    Exactly the two messages the current path produces via ``split_chat_prompt``: the system text
    is the text before the first runtime-data heading, the user message is the runtime data, a
    blank line, ``User message:`` and the user's text. History is NOT used (today it is already
    inside the ``conversation_trace`` section text) and no caps are applied.

``chat_roles`` (lab X9)
    ``[system(header only), (user, assistant) * N oldest -> newest, final user]``. All state
    sections form ONE labelled quoted-data block (``RUNTIME_DATA_HEADING`` line, then the sections
    separated by blank lines) attached to the FIRST user turn as
    ``block + "\\n\\nUser message:\\n" + text``, mirroring ``experiments/r20/bare_chat_adapter.compose_first_user``. With history the block is
    on the oldest kept exchange's user turn and the final user message is the user's text only;
    without history the block is on the only user turn (the final one), as in the lab adapter.
    Deviation from the lab: the lab sent a short bare system line and put the real header inside
    the block; here the real header is the system message, as the Issue's proposal says. That exact
    variant has NOT been run in the lab.

``chat_roles_split`` (lab X10)
    Like ``chat_roles`` but sections with origin ``code`` follow the header in the system message
    (``header + "\\n\\n" + section`` ...) and only ``stored`` sections go into the block.

Invariant (#414): the system message is built only from ``system_header`` and ``code``-origin
sections (the latter only in ``chat_roles_split``). Stored sections, history and the user text are
never routed to it; there is no code path from them to the system role.

Caps drop whole units and never truncate: oldest exchanges first (also beyond
``max_exchanges``), and, if the state block exceeds ``max_state_block_bytes``, whole sections from
the END of the block. An assistant text is never shortened. The user's own text is never capped.
Caps are not applied in ``legacy`` (exact reproduction).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Literal

from suzka.persona.prompt_builder import CURRENT_MESSAGE_HEADING, RUNTIME_DATA_HEADING

# DEFAULTS the Human has not fixed yet (#500 "What a human must decide").
DEFAULT_MAX_EXCHANGES = 6
DEFAULT_MAX_HISTORY_BYTES = 32768
DEFAULT_MAX_STATE_BLOCK_BYTES = 32768
DEFAULT_MODE = "legacy"
_BOUND_MAX_EXCHANGES = 1000
_BOUND_MAX_BYTES = 16 * 1024 * 1024

_USER_MESSAGE_MARK = f"\n\n{CURRENT_MESSAGE_HEADING}\n"
_Role = Literal["system", "user", "assistant"]
_Origin = Literal["code", "stored"]


class CompositionMode(str, Enum):
    """Composition modes. The names are defaults, not fixed by the Human."""

    LEGACY = "legacy"
    CHAT_ROLES = "chat_roles"
    CHAT_ROLES_SPLIT = "chat_roles_split"


def _nbytes(text: str) -> int:
    # surrogatepass: a lone surrogate counts as 3 bytes (conservative) instead of raising.
    return len(text.encode("utf-8", "surrogatepass"))


def _require_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a str")
    return value


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: _Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("role must be system, user or assistant")
        _require_str(self.content, "content")


@dataclass(frozen=True, slots=True)
class StateSection:
    """One runtime-state section. ``origin`` says whether code generated the text or it may carry
    user- or model-originated stored text. ``lead`` is the newline run that precedes the section
    in the legacy layout (only ``legacy`` uses it; chat modes join sections with a blank line)."""

    name: str
    text: str
    origin: _Origin
    lead: str = "\n\n"

    def __post_init__(self) -> None:
        if not _require_str(self.name, "name"):
            raise ValueError("name must not be empty")
        _require_str(self.text, "text")
        if self.origin not in ("code", "stored"):
            raise ValueError("origin must be code or stored")
        if _require_str(self.lead, "lead").strip("\n"):
            raise ValueError("lead must consist of newlines only")


@dataclass(frozen=True, slots=True)
class HistoryExchange:
    user_text: str
    assistant_text: str

    def __post_init__(self) -> None:
        _require_str(self.user_text, "user_text")
        _require_str(self.assistant_text, "assistant_text")


@dataclass(frozen=True, slots=True)
class CompositionInput:
    system_header: str
    state_sections: tuple[StateSection, ...]
    history: tuple[HistoryExchange, ...]
    user_text: str

    def __post_init__(self) -> None:
        header = _require_str(self.system_header, "system_header")
        if RUNTIME_DATA_HEADING in header:
            raise ValueError("system_header must not contain the runtime-data heading")
        _require_str(self.user_text, "user_text")
        if type(self.state_sections) is not tuple or not all(
            type(item) is StateSection for item in self.state_sections
        ):
            raise TypeError("state_sections must be a tuple of StateSection")
        if type(self.history) is not tuple or not all(
            type(item) is HistoryExchange for item in self.history
        ):
            raise TypeError("history must be a tuple of HistoryExchange")


def _require_bound(value: object, label: str, upper: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an int")
    if not 0 <= value <= upper:
        raise ValueError(f"{label} must be between 0 and {upper}")
    return value


@dataclass(frozen=True, slots=True)
class CompositionCaps:
    """Bounds. Every default is provisional (not fixed by the Human)."""

    max_exchanges: int = DEFAULT_MAX_EXCHANGES
    max_history_bytes: int = DEFAULT_MAX_HISTORY_BYTES
    max_state_block_bytes: int = DEFAULT_MAX_STATE_BLOCK_BYTES

    def __post_init__(self) -> None:
        _require_bound(self.max_exchanges, "max_exchanges", _BOUND_MAX_EXCHANGES)
        _require_bound(self.max_history_bytes, "max_history_bytes", _BOUND_MAX_BYTES)
        _require_bound(self.max_state_block_bytes, "max_state_block_bytes", _BOUND_MAX_BYTES)


@dataclass(frozen=True, slots=True)
class CompositionReport:
    """Numbers only (no text), so wiring can log them."""

    mode: str
    exchanges_used: int
    exchanges_dropped: int
    state_block_bytes: int
    system_bytes: int
    sections_dropped: int = 0


def _mode_of(mode: CompositionMode | str) -> CompositionMode:
    try:
        return CompositionMode(mode)
    except ValueError:
        raise ValueError(f"unknown composition mode: {mode!r}") from None


def _window(history: tuple[HistoryExchange, ...], caps: CompositionCaps) -> tuple[HistoryExchange, ...]:
    """Most recent exchanges within both caps. Whole exchanges only; oldest dropped first."""

    kept = list(history[max(0, len(history) - caps.max_exchanges) :] if caps.max_exchanges else ())
    total = sum(_nbytes(item.user_text) + _nbytes(item.assistant_text) for item in kept)
    while kept and total > caps.max_history_bytes:
        oldest = kept.pop(0)
        total -= _nbytes(oldest.user_text) + _nbytes(oldest.assistant_text)
    return tuple(kept)


def _block(sections: list[StateSection], caps: CompositionCaps) -> tuple[str, int]:
    """(quoted-data block or "", sections dropped). Empty sections are skipped; whole sections
    are dropped from the end until the block fits."""

    texts = [section.text for section in sections if section.text]
    dropped = 0
    while texts:
        block = RUNTIME_DATA_HEADING + "\n" + "\n\n".join(texts)
        if _nbytes(block) <= caps.max_state_block_bytes:
            return block, dropped
        texts.pop()
        dropped += 1
    return "", dropped


def compose_with_report(
    mode: CompositionMode | str,
    inp: CompositionInput,
    caps: CompositionCaps | None = None,
) -> tuple[tuple[ChatMessage, ...], CompositionReport]:
    """The chat messages for ``mode`` and a numbers-only report."""

    selected = _mode_of(mode)
    if type(inp) is not CompositionInput:
        raise TypeError("inp must be a CompositionInput")
    bounds = CompositionCaps() if caps is None else caps
    if type(bounds) is not CompositionCaps:
        raise TypeError("caps must be a CompositionCaps")

    if selected is CompositionMode.LEGACY:
        runtime = RUNTIME_DATA_HEADING + "\n" + "".join(s.lead + s.text for s in inp.state_sections)
        messages = (
            ChatMessage("system", inp.system_header),
            ChatMessage("user", runtime + _USER_MESSAGE_MARK + inp.user_text),
        )
        return messages, CompositionReport(
            mode=selected.value, exchanges_used=0, exchanges_dropped=0,
            state_block_bytes=_nbytes(runtime), system_bytes=_nbytes(inp.system_header),
        )

    split = selected is CompositionMode.CHAT_ROLES_SPLIT
    # The only inputs that can reach the system role: the header and (split mode) code-origin sections.
    system_sections = [s for s in inp.state_sections if split and s.origin == "code"]
    block_sections = [s for s in inp.state_sections if not (split and s.origin == "code")]
    system_text = inp.system_header + "".join("\n\n" + s.text for s in system_sections if s.text)
    block, sections_dropped = _block(block_sections, bounds)
    window = _window(inp.history, bounds)

    out: list[ChatMessage] = [ChatMessage("system", system_text)]
    for index, exchange in enumerate(window):
        user = exchange.user_text
        if index == 0 and block:
            user = block + _USER_MESSAGE_MARK + user
        out.append(ChatMessage("user", user))
        out.append(ChatMessage("assistant", exchange.assistant_text))
    final = inp.user_text
    if not window and block:
        final = block + _USER_MESSAGE_MARK + final
    out.append(ChatMessage("user", final))
    return tuple(out), CompositionReport(
        mode=selected.value, exchanges_used=len(window),
        exchanges_dropped=len(inp.history) - len(window),
        state_block_bytes=_nbytes(block), system_bytes=_nbytes(system_text),
        sections_dropped=sections_dropped,
    )


def compose(
    mode: CompositionMode | str,
    inp: CompositionInput,
    caps: CompositionCaps | None = None,
) -> tuple[ChatMessage, ...]:
    return compose_with_report(mode, inp, caps)[0]


# --------------------------------------------------------------------------- from a built prompt

_ID = r"[A-Za-z0-9_.:\-]+"
_CONTEXT_SHAPE = re.compile(
    rf"Current Context:\n- context_id: {_ID}\n- context_type: {_ID}\n- source_channel: {_ID}\n"
    rf"- source_session_id: {_ID}\n- participant_refs: {_ID}(?:, {_ID})*",
    re.ASCII,
)
_NUM = r"-?[0-9]+\.[0-9]{6}"
_EMOTION_SHAPE = re.compile(
    rf"Emotion:\n- valence: {_NUM}\n- arousal: {_NUM}\n- optimal_loss: {_NUM}", re.ASCII,
)
_NEWLINE_RUN = re.compile(r"(\n{2,})")


def _shape(chunk: str) -> str | None:
    if _CONTEXT_SHAPE.fullmatch(chunk):
        return "context"
    if _EMOTION_SHAPE.fullmatch(chunk):
        return "emotion"
    return None


def _classify(chunk: str, index: int, unique_shapes: frozenset[str]) -> tuple[str, _Origin]:
    """Name and origin of one runtime-data chunk. Shape alone cannot prove that code generated a
    chunk (a stored trace line can imitate it), so ``code`` needs a strict character-level shape AND
    that exactly one chunk in the whole runtime data has that shape; a duplicated shape makes every
    copy ``stored``. Anything else is ``stored`` (conservative)."""

    shape = _shape(chunk)
    if shape is not None and shape in unique_shapes:
        return shape, "code"
    if chunk.split("\n", 1)[0] == "Active Values:":
        return "values", "stored"
    return f"section_{index}", "stored"


def composition_input_from_split(
    system_text: str,
    user_part: str,
    user_input: str,
    history: tuple[HistoryExchange, ...] = (),
) -> CompositionInput | None:
    """Rebuild a ``CompositionInput`` from the ``(system, user)`` pair of ``split_chat_prompt``.

    Lossless for ``legacy``: ``compose("legacy", ...)`` returns exactly that pair. Returns None if
    ``user_part`` is not ``RUNTIME_DATA_HEADING``, runtime data and the ``User message:`` suffix for
    ``user_input`` or ``system_text`` contains the heading. Runtime data is cut at blank lines
    into sections; a Context or Emotion chunk is ``code`` origin only with the strict builder shape
    and only if exactly one chunk has that shape (a forged copy in stored text makes both ``stored``).
    """

    suffix = _USER_MESSAGE_MARK + user_input
    prefix = RUNTIME_DATA_HEADING + "\n"
    if not (user_part.startswith(prefix) and user_part.endswith(suffix)):
        return None
    if len(user_part) < len(prefix) + len(suffix):
        return None
    rest = user_part[len(prefix) : len(user_part) - len(suffix)]
    leading = len(rest) - len(rest.lstrip("\n"))
    parts = [rest[:leading], *_NEWLINE_RUN.split(rest[leading:])]
    # parts: lead0, chunk0, sep1, chunk1, sep2, ...
    chunks = [parts[position] for position in range(1, len(parts), 2)]
    shapes = [shape for shape in map(_shape, chunks) if shape is not None]
    unique = frozenset(shape for shape in shapes if shapes.count(shape) == 1)
    sections: list[StateSection] = []
    for position in range(1, len(parts), 2):
        chunk = parts[position]
        lead = parts[0] if position == 1 else parts[position - 1]
        name, origin = _classify(chunk, len(sections), unique)
        sections.append(StateSection(name, chunk, origin, lead))
    if RUNTIME_DATA_HEADING in system_text:
        return None
    return CompositionInput(system_text, tuple(sections), history, user_input)
