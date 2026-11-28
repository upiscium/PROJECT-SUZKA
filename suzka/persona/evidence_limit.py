"""Pure per-item limits on the WorkingMemory evidence rows of the model INPUT text (#500 slice 3b-4).

Owns: how the rendered Attention text is limited before it is sent to the model: keep the first ``top_k``
WorkingMemory evidence rows in rendered order (no re-ranking) and cut each kept row's ``text`` to at most
``item_max_chars`` characters (the ``…`` marker counts inside the limit).
May: transform a COPY of the rendered text / of one ``PromptSections`` value; return counts.
Must not: touch the Attention payload, its digest or byte witness, the builder, the Motivation / Goal /
Commitment rows, any heading or the authority intro, or the legacy ``stored_*_evidence`` sections; select or
rerank anything; return or log row text; import runtime, models or main_loop.

Parsing is strict and FAILS CLOSED: on any doubt the input is returned unchanged and the numbers carry a skip
reason label.  Headings come from ``suzka.attention.policy`` and rows are written with the repository's own
``canonical_json``, so a cut row is always one valid canonical JSON line (a JSON escape is never split because
the cut is made on the decoded text).
"""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Final

from suzka.attention.common import canonical_json
from suzka.attention.policy import ATTENTION_PROMPT_AUTHORITY_INTRO, ATTENTION_PROMPT_SECTION_HEADERS
from suzka.persona.prompt_builder import PromptSections

ELLIPSIS: Final[str] = "…"
EVIDENCE_TOP_K_RANGE: Final[tuple[int, int]] = (1, 50)
EVIDENCE_ITEM_CHARS_RANGE: Final[tuple[int, int]] = (20, 2000)

ATTENTION_SECTION_NAME: Final[str] = "attention"
_LEGACY_EVIDENCE_SECTIONS: Final[frozenset[str]] = frozenset(
    {"stored_episodic_evidence", "stored_semantic_evidence"}
)
_WM_ROW_KEYS: Final[frozenset[str]] = frozenset({"source_kind", "text"})
# The first renderer heading is the WorkingMemory evidence heading (policy order); the rest follow it.
_HEADING_LINES: Final[tuple[str, ...]] = tuple(
    heading.rstrip("\n") for _, heading in ATTENTION_PROMPT_SECTION_HEADERS
)

SKIP_LIMITS_OFF: Final[str] = "limits_off"
SKIP_INTRO: Final[str] = "attention_intro"
SKIP_TEXT_END: Final[str] = "attention_text_end"
SKIP_HEADING_ORDER: Final[str] = "attention_heading_order"
SKIP_HEADING: Final[str] = "wm_evidence_heading"
SKIP_NOT_JSON: Final[str] = "wm_evidence_row_not_json"
SKIP_SHAPE: Final[str] = "wm_evidence_row_shape"
SKIP_NOT_CANONICAL: Final[str] = "wm_evidence_row_not_canonical"
SKIP_NO_ATTENTION: Final[str] = "attention_section_missing"
SKIP_AMBIGUOUS: Final[str] = "attention_section_ambiguous"
SKIP_LEGACY: Final[str] = "legacy_evidence_section"

# Closed vocabulary of every skip label (``numbers["skipped"]`` is None or one of these).
SKIP_REASONS: Final[frozenset[str]] = frozenset(
    {
        SKIP_LIMITS_OFF, SKIP_INTRO, SKIP_TEXT_END, SKIP_HEADING_ORDER, SKIP_HEADING, SKIP_NOT_JSON,
        SKIP_SHAPE, SKIP_NOT_CANONICAL, SKIP_NO_ATTENTION, SKIP_AMBIGUOUS, SKIP_LEGACY,
    }
)

Numbers = dict[str, int | str | None]


def _check_bound(name: str, value: int | None, bounds: tuple[int, int]) -> None:
    if value is None:
        return
    if type(value) is not int or not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} must be None or an integer in {bounds[0]}..{bounds[1]}")


def _numbers(text: str, *, skipped: str | None, rows: int | None = None) -> Numbers:
    return {
        "rows_before": rows, "rows_after": rows, "chars_before": len(text), "chars_after": len(text),
        "evidence_chars_before": None, "evidence_chars_after": None, "skipped": skipped,
    }


def _split(text: str) -> tuple[list[str], list[str], list[str]] | str:
    """(the WM heading line, the WM row lines, the lines from the next heading on) or a skip reason."""

    if not text.startswith(ATTENTION_PROMPT_AUTHORITY_INTRO):
        return SKIP_INTRO
    if not text.endswith("\n"):
        return SKIP_TEXT_END
    lines = text[len(ATTENTION_PROMPT_AUTHORITY_INTRO):-1].split("\n")
    positions: list[int] = []
    for number, heading in enumerate(_HEADING_LINES):
        found = [index for index, line in enumerate(lines) if line == heading]
        if len(found) != 1:
            return SKIP_HEADING if number == 0 else SKIP_HEADING_ORDER
        positions.append(found[0])
    if positions[0] != 0:
        return SKIP_HEADING
    if positions != sorted(positions):
        return SKIP_HEADING_ORDER
    return lines[:1], lines[1:positions[1]], lines[positions[1]:]


def _parse_rows(row_lines: list[str]) -> tuple[list[dict[str, str]], str | None]:
    rows: list[dict[str, str]] = []
    for line in row_lines:
        try:
            row = json.loads(line)
        except (ValueError, RecursionError):  # deeply nested JSON must fail closed, not escape
            return [], SKIP_NOT_JSON
        if (
            type(row) is not dict or set(row) != _WM_ROW_KEYS
            or not all(type(value) is str for value in row.values())
        ):
            return [], SKIP_SHAPE
        try:
            canonical = canonical_json(row).decode("ascii") == line
        except (ValueError, UnicodeError):
            canonical = False
        if not canonical:
            return [], SKIP_NOT_CANONICAL
        rows.append(row)
    return rows, None


def limit_attention_text(text: str, top_k: int | None, item_max_chars: int | None) -> tuple[str, Numbers]:
    """(the rendered Attention text with the WM evidence rows limited, numbers).

    ``ValueError`` only for an out-of-range limit; odd text is a skip (unchanged text + a reason label).
    ``numbers`` holds counts and a skip label (one of ``SKIP_REASONS``), never text.

    Label priority: limit validation first, then ``limits_off`` (both limits None; no parsing at all), then
    the structural labels in text order (intro, trailing newline, headings, rows).

    The limit is on DECODED characters, not an input-size guarantee: output rows are canonical ASCII JSON, so
    the ``…`` marker is written as the 6-byte escape ``\\u2026``.  A row of ``item_max_chars + 1`` ASCII
    characters therefore gets 4 bytes LONGER, and ``chars_after`` can exceed ``chars_before``.
    """

    _check_bound("top_k", top_k, EVIDENCE_TOP_K_RANGE)
    _check_bound("item_max_chars", item_max_chars, EVIDENCE_ITEM_CHARS_RANGE)
    if type(text) is not str:
        raise TypeError("text must be a string")
    if top_k is None and item_max_chars is None:
        return text, _numbers(text, skipped=SKIP_LIMITS_OFF)
    parts = _split(text)
    if isinstance(parts, str):
        return text, _numbers(text, skipped=parts)
    head, row_lines, tail = parts
    rows, reason = _parse_rows(row_lines)
    if reason is not None:
        return text, _numbers(text, skipped=reason)
    kept = rows if top_k is None else rows[:top_k]
    if item_max_chars is not None:
        kept = [
            row if len(row["text"]) <= item_max_chars
            else {**row, "text": row["text"][: item_max_chars - 1] + ELLIPSIS}
            for row in kept
        ]
    new_rows = [canonical_json(row).decode("ascii") for row in kept]
    new_text = ATTENTION_PROMPT_AUTHORITY_INTRO + "\n".join([*head, *new_rows, *tail]) + "\n"
    numbers = _numbers(text, skipped=None, rows=len(rows))
    numbers.update(
        rows_after=len(kept), chars_after=len(new_text),
        evidence_chars_before=sum(len(line) + 1 for line in row_lines),
        evidence_chars_after=sum(len(line) + 1 for line in new_rows),
    )
    return new_text, numbers


def limit_sections(
    sections: PromptSections, top_k: int | None, item_max_chars: int | None
) -> tuple[PromptSections, Numbers]:
    """``sections`` with ONLY the ``attention`` section's text limited; all else (and every origin) untouched."""

    _check_bound("top_k", top_k, EVIDENCE_TOP_K_RANGE)
    _check_bound("item_max_chars", item_max_chars, EVIDENCE_ITEM_CHARS_RANGE)
    if top_k is None and item_max_chars is None:
        return sections, _numbers("", skipped=SKIP_LIMITS_OFF)  # same priority as limit_attention_text
    names = [section.name for section in sections.sections]
    attention = [index for index, name in enumerate(names) if name == ATTENTION_SECTION_NAME]
    if len(attention) != 1:
        if attention:
            reason = SKIP_AMBIGUOUS
        elif _LEGACY_EVIDENCE_SECTIONS.intersection(names):
            reason = SKIP_LEGACY
        else:
            reason = SKIP_NO_ATTENTION
        return sections, _numbers("", skipped=reason)
    index = attention[0]
    old = sections.sections[index]
    new_text, numbers = limit_attention_text(old.text, top_k, item_max_chars)
    if numbers["skipped"] is not None or new_text == old.text:
        return sections, numbers
    return replace(
        sections, sections=(*sections.sections[:index], replace(old, text=new_text), *sections.sections[index + 1:])
    ), numbers
