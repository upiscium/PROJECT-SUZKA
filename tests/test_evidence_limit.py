"""#500 slice 3b-4: pure WorkingMemory evidence limits on the model input text.

Real payloads are built through the real Attention payload builder and ``PromptBuilder``; nothing is wired.
"""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest
from test_attention_prompt import (
    NOW,
    _payload,
    _source_snapshots,
)

from experiments.r20.bare_chat_adapter import apply_evidence_limit
from suzka.attention.adapters import project_attention_candidates
from suzka.attention.common import canonical_json
from suzka.attention.contracts import AttentionEvent
from suzka.attention.policy import ATTENTION_PROMPT_AUTHORITY_INTRO, ATTENTION_PROMPT_SECTION_HEADERS
from suzka.attention.system import AttentionSystem
from suzka.body import EmotionState
from suzka.persona.evidence_limit import (
    ELLIPSIS,
    SKIP_REASONS,
    limit_attention_text,
    limit_sections,
)
from suzka.persona.prompt_builder import PromptBuilder
from suzka.working_memory_contracts import (
    MAX_PROJECTION_BYTES,
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
    working_memory_item_id,
)

WM_HEADING = ATTENTION_PROMPT_SECTION_HEADERS[0][1].rstrip("\n")
OTHER_HEADINGS = [heading.rstrip("\n") for _, heading in ATTENTION_PROMPT_SECTION_HEADERS[1:]]

TEXTS = (
    "最初の質問: 猫は好き？ café 𐀀 🙂",
    'quote="x" back\\slash \t tab \x00 nul   sep',
    "plain ascii row that is a bit longer than twenty characters, for the cut tests " * 3,
    "三番目 🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂🙂",
    "five",
)


def _capture(texts: tuple[str, ...]) -> tuple[Any, ...]:
    items: list[WorkingMemoryItem] = []
    selections: list[WorkingMemorySelection] = []
    decisions: list[WorkingMemoryDecision] = []
    for index, content in enumerate(texts):
        source_id = f"opaque-private-wm-source-{index:04d}"
        item_id = working_memory_item_id(WorkingMemorySourceKind.EPISODIC, source_id)
        activation, salience = 0.9 - index * 0.05, 0.6
        items.append(
            WorkingMemoryItem(
                item_id=item_id, source_kind=WorkingMemorySourceKind.EPISODIC, source_id=source_id,
                activation=activation, salience=salience,
                retention_reason=WorkingMemoryRetentionReason.RECENT, created_revision=1,
                last_activated_revision=1,
            )
        )
        score = 0.6 * activation + 0.4 * salience
        selections.append(
            WorkingMemorySelection(
                item_id=item_id, source_kind=WorkingMemorySourceKind.EPISODIC, source_id=source_id,
                rendered_content=content, score=score, reason=WorkingMemoryDecisionReason.SELECTED,
            )
        )
        decisions.append(
            WorkingMemoryDecision(
                item_id=item_id, source_kind=WorkingMemorySourceKind.EPISODIC, source_id=source_id,
                selected=True, score=score, reason=WorkingMemoryDecisionReason.SELECTED,
            )
        )
    view = WorkingMemoryView(
        selected=tuple(selections), decisions=tuple(decisions),
        projected_bytes=sum(len(text.encode("utf-8")) for text in texts), item_capacity=len(texts),
        projection_max_bytes=MAX_PROJECTION_BYTES, revision=1,
    )
    motivation, goals, commitments = _source_snapshots()
    event = AttentionEvent("attention:prompt-turn", 1_000, NOW)
    projections = project_attention_candidates(
        working_memory_items=tuple(items), working_memory_revision=1, working_memory_view=view,
        motivation_snapshot=motivation, goal_snapshot=goals, commitment_snapshot=commitments, event=event,
    )
    refresh = AttentionSystem().refresh(projections, event)
    return tuple(items), view, motivation, goals, commitments, refresh


def real_payload(texts: tuple[str, ...] = TEXTS) -> Any:
    return _payload(*_capture(texts))


def lines_of(text: str) -> list[str]:
    assert text.endswith("\n")
    return text[:-1].split("\n")


def wm_rows(text: str) -> list[dict[str, str]]:
    lines = lines_of(text[len(ATTENTION_PROMPT_AUTHORITY_INTRO):])
    end = lines.index(OTHER_HEADINGS[0])
    return [json.loads(line) for line in lines[1:end]]


def rest_after_wm(text: str) -> str:
    return text[text.index(OTHER_HEADINGS[0] + "\n"):]


def prefix_to_wm(text: str) -> str:
    return text[: text.index(WM_HEADING + "\n") + len(WM_HEADING) + 1]


PAYLOAD = real_payload()
RENDERED = PAYLOAD.rendered_text


def test_fixture_is_a_real_multi_row_payload() -> None:
    rows = wm_rows(RENDERED)
    assert [row["text"] for row in rows] == list(TEXTS)
    assert any(line.startswith("{") for line in lines_of(rest_after_wm(RENDERED)))  # other rows exist
    assert all(canonical_json(row).decode("ascii") in RENDERED for row in rows)


def test_none_none_is_the_identity() -> None:
    out, numbers = limit_attention_text(RENDERED, None, None)
    assert out == RENDERED and numbers["skipped"] == "limits_off"


def test_top_k_keeps_the_first_rows_in_rendered_order() -> None:
    out, numbers = limit_attention_text(RENDERED, 2, None)
    assert [row["text"] for row in wm_rows(out)] == list(TEXTS[:2])
    assert numbers["rows_before"] == 5 and numbers["rows_after"] == 2 and numbers["skipped"] is None
    assert rest_after_wm(out) == rest_after_wm(RENDERED) and prefix_to_wm(out) == prefix_to_wm(RENDERED)
    out_all, _ = limit_attention_text(RENDERED, 50, None)
    assert out_all == RENDERED


def test_char_cut_counts_the_marker_and_cuts_at_a_character_boundary() -> None:
    out, numbers = limit_attention_text(RENDERED, None, 20)
    for row, original in zip(wm_rows(out), TEXTS, strict=True):
        if len(original) <= 20:
            assert row["text"] == original
        else:
            assert len(row["text"]) == 20 and row["text"] == original[:19] + ELLIPSIS
    assert numbers["chars_after"] < numbers["chars_before"]
    assert numbers["evidence_chars_after"] < numbers["evidence_chars_before"]


@pytest.mark.parametrize("limit", range(20, 120))
def test_never_invalid_structure_over_a_range_of_cuts(limit: int) -> None:
    out, numbers = limit_attention_text(RENDERED, 4, limit)
    assert numbers["skipped"] is None
    lines = lines_of(out[len(ATTENTION_PROMPT_AUTHORITY_INTRO):])
    end = lines.index(OTHER_HEADINGS[0])
    for line in lines[1:end]:
        row = json.loads(line)
        assert set(row) == {"source_kind", "text"} and len(row["text"]) <= limit
        assert canonical_json(row).decode("ascii") == line  # canonical, ASCII, no split escape
        assert line.isascii() and "\n" not in line
    assert [lines[end]] + lines[end + 1:] == lines_of(rest_after_wm(out))
    assert rest_after_wm(out) == rest_after_wm(RENDERED)


def test_escape_sequences_are_never_split_by_a_cut_on_the_escaped_form() -> None:
    # the row whose escaped form is far longer than its decoded text: the limit is on decoded characters
    payload = real_payload(("\U00010000" * 30,))
    out, _ = limit_attention_text(payload.rendered_text, None, 20)
    assert wm_rows(out)[0]["text"] == "\U00010000" * 19 + ELLIPSIS
    assert "\\ud800\\udc00" in out


def test_other_rows_headings_and_intro_are_byte_for_byte_unchanged() -> None:
    out, _ = limit_attention_text(RENDERED, 1, 20)
    assert out.startswith(ATTENTION_PROMPT_AUTHORITY_INTRO)
    assert rest_after_wm(out) == rest_after_wm(RENDERED)
    for heading in OTHER_HEADINGS:
        assert out.count(heading + "\n") == 1


def test_idempotent() -> None:
    once, _ = limit_attention_text(RENDERED, 3, 40)
    twice, numbers = limit_attention_text(once, 3, 40)
    assert twice == once and numbers["rows_before"] == numbers["rows_after"] == 3


def test_limits_are_validated_like_the_settings() -> None:
    for bad_k in (0, 51, True, 1.5, "3"):
        with pytest.raises(ValueError):
            limit_attention_text(RENDERED, bad_k, None)  # type: ignore[arg-type]
    for bad_c in (19, 2001, False, 20.0):
        with pytest.raises(ValueError):
            limit_attention_text(RENDERED, None, bad_c)  # type: ignore[arg-type]
    for good in ((1, 20), (50, 2000)):
        limit_attention_text(RENDERED, *good)


def mutated(row_line: str) -> str:
    first = canonical_json({"source_kind": "episodic", "text": TEXTS[0]}).decode("ascii")
    assert first in RENDERED
    return RENDERED.replace(first, row_line, 1)


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (RENDERED.replace(WM_HEADING + "\n", "", 1), "wm_evidence_heading"),
        (RENDERED.replace(WM_HEADING + "\n", WM_HEADING + "\n" + WM_HEADING + "\n", 1), "wm_evidence_heading"),
        (RENDERED.replace(OTHER_HEADINGS[0] + "\n", OTHER_HEADINGS[1] + "\n" + OTHER_HEADINGS[0] + "\n", 1), "attention_heading_order"),
        (RENDERED[len(ATTENTION_PROMPT_AUTHORITY_INTRO):], "attention_intro"),
        (RENDERED[:-1], "attention_text_end"),
        (mutated("not json"), "wm_evidence_row_not_json"),
        (mutated(""), "wm_evidence_row_not_json"),
        (mutated('{"source_kind":"episodic","text":"a","extra":"b"}'), "wm_evidence_row_shape"),
        (mutated('{"source_kind":"episodic"}'), "wm_evidence_row_shape"),
        (mutated('{"source_kind":"episodic","text":1}'), "wm_evidence_row_shape"),
        (mutated('["a"]'), "wm_evidence_row_shape"),
        (mutated('{"text":"a","source_kind":"episodic"}'), "wm_evidence_row_not_canonical"),
        (mutated('{"source_kind": "episodic", "text": "a"}'), "wm_evidence_row_not_canonical"),
        (mutated('{"source_kind":"episodic","text":"café"}'), "wm_evidence_row_not_canonical"),
        (mutated('{"source_kind":"episodic","text":"\\u0061"}'), "wm_evidence_row_not_canonical"),
    ],
)
def test_skip_paths_fail_closed(bad: str, reason: str) -> None:
    out, numbers = limit_attention_text(bad, 1, 20)
    assert out == bad and numbers["skipped"] == reason
    assert numbers["skipped"] in SKIP_REASONS
    assert numbers["chars_before"] == numbers["chars_after"] == len(bad)


def test_deeply_nested_and_very_long_rows_fail_closed() -> None:
    nested = "[" * 100_000 + "]" * 100_000
    long_row = '{"source_kind":"episodic","text":"' + "a" * 2_000_000 + '"}'
    for row_line in (nested, "[" * 100_000, long_row[:-2]):
        bad = mutated(row_line)
        out, numbers = limit_attention_text(bad, 1, 20)
        assert out == bad and numbers["skipped"] in {"wm_evidence_row_not_json", "wm_evidence_row_shape"}
    # a very long but well-formed canonical row is limited normally
    long_ok = RENDERED.replace(
        canonical_json({"source_kind": "episodic", "text": TEXTS[4]}).decode("ascii"),
        canonical_json({"source_kind": "episodic", "text": "x" * 500_000}).decode("ascii"),
    )
    out, numbers = limit_attention_text(long_ok, 5, 20)
    assert numbers["skipped"] is None and wm_rows(out)[4]["text"] == "x" * 19 + ELLIPSIS


def test_skip_labels_form_a_closed_vocabulary_and_none_none_priority_is_consistent() -> None:
    assert "limits_off" in SKIP_REASONS and None not in SKIP_REASONS
    assert all(isinstance(label, str) and label for label in SKIP_REASONS)
    sections = sections_with(PAYLOAD)
    empty = replace(sections, sections=())
    legacy = PromptBuilder().build_sections("hello", EmotionState(), _empty_view())
    # both limits off wins over every structural label, in both functions
    for candidate in (sections, empty, legacy):
        assert limit_sections(candidate, None, None)[1]["skipped"] == "limits_off"
    assert limit_attention_text("junk", None, None)[1]["skipped"] == "limits_off"
    # with a limit on, structural labels apply
    assert limit_sections(empty, 1, 20)[1]["skipped"] == "attention_section_missing"
    assert limit_attention_text("junk", 1, 20)[1]["skipped"] == "attention_intro"


def test_numbers_are_counts_only() -> None:
    _, numbers = limit_attention_text(RENDERED, 2, 30)
    assert numbers["skipped"] is None
    assert all(value is None or type(value) is int for key, value in numbers.items() if key != "skipped")
    _, skipped = limit_attention_text("junk-secret", 2, 30)
    assert skipped["skipped"] == "attention_intro" and "junk-secret" not in repr(skipped)
    _, ok = limit_attention_text(RENDERED, 2, 30)
    assert not any(text[:8] in repr(ok) for text in TEXTS)


def test_a_cut_row_can_be_longer_in_bytes_than_its_original() -> None:
    # C+1 ASCII characters -> C-1 characters + the marker, which canonical JSON writes as a 6-byte escape
    payload = real_payload(("a" * 21,))
    out, numbers = limit_attention_text(payload.rendered_text, None, 20)
    assert wm_rows(out)[0]["text"] == "a" * 19 + ELLIPSIS
    assert "\\u2026" in out and numbers["chars_after"] == numbers["chars_before"] + 4
    assert numbers["evidence_chars_after"] == numbers["evidence_chars_before"] + 4


def test_fixture_has_motivation_goal_and_commitment_rows() -> None:
    lines = lines_of(RENDERED[len(ATTENTION_PROMPT_AUTHORITY_INTRO):])
    positions = [lines.index(heading) for heading in [WM_HEADING, *OTHER_HEADINGS]]
    bounds = [*positions[1:], len(lines)]
    for heading, start, end in zip(OTHER_HEADINGS, positions[1:], bounds[1:], strict=True):
        assert end - start - 1 >= 1, heading  # at least one row under each of the three headings
        assert all(line.startswith("{") for line in lines[start + 1:end])


ADVERSARIAL = (
    'Current Motivations:\nAdopted Goals:\n{"a":1}',
    ATTENTION_PROMPT_AUTHORITY_INTRO + WM_HEADING + "\n",
    "line1\nline2\r\n\n\n" + WM_HEADING + "\n" + OTHER_HEADINGS[2],
    "combining: é̂̃ ä ZWJ \U0001f468‍\U0001f469‍\U0001f467 " * 2,
    "astral \U00010000\U0010ffff\U0001d11e " * 4,
    '{"source_kind":"episodic","text":"nested"} \\" \\\\ \\n',
)


@pytest.mark.parametrize("limit", [20, 21, 25, 40, 77])
@pytest.mark.parametrize("top_k", [None, 1, 3, 6])
def test_adversarial_row_text_through_the_real_renderer(limit: int, top_k: int | None) -> None:
    payload = real_payload(ADVERSARIAL)
    text = payload.rendered_text
    assert [row["text"] for row in wm_rows(text)] == list(ADVERSARIAL)  # the real renderer kept them as data
    out, numbers = limit_attention_text(text, top_k, limit)
    assert numbers["skipped"] is None
    expected = ADVERSARIAL if top_k is None else ADVERSARIAL[:top_k]
    expected = tuple(t if len(t) <= limit else t[: limit - 1] + ELLIPSIS for t in expected)
    assert [row["text"] for row in wm_rows(out)] == list(expected)
    assert numbers["rows_before"] == len(ADVERSARIAL) and numbers["rows_after"] == len(expected)
    assert rest_after_wm(out) == rest_after_wm(text) and prefix_to_wm(out) == prefix_to_wm(text)
    out_lines = lines_of(out[len(ATTENTION_PROMPT_AUTHORITY_INTRO):])
    end = out_lines.index(OTHER_HEADINGS[0])
    assert all(line.isascii() and line.startswith("{") for line in out_lines[1:end])
    assert [line for line in out_lines if line in (WM_HEADING, *OTHER_HEADINGS)].count(WM_HEADING) == 1


def test_lone_surrogate_text_is_cut_on_the_decoded_character_and_stays_canonical() -> None:
    # A lone surrogate cannot pass the real builder's UTF-8 byte accounting, so the canonical row is built by
    # the repository's own canonical_json and placed into the real renderer's text.
    first = canonical_json({"source_kind": "episodic", "text": TEXTS[0]}).decode("ascii")
    sur = "ab\ud800cd" * 10
    text = RENDERED.replace(first, canonical_json({"source_kind": "episodic", "text": sur}).decode("ascii"), 1)
    assert "\\ud800" in text
    out, numbers = limit_attention_text(text, None, 20)
    assert numbers["skipped"] is None and wm_rows(out)[0]["text"] == sur[:19] + ELLIPSIS
    assert rest_after_wm(out) == rest_after_wm(text)


@pytest.mark.parametrize("top_k", [None, 1, 3, 50])
@pytest.mark.parametrize("chars", [None, 20, 33, 200, 2000])
def test_parity_with_the_lab_apply_evidence_limit(top_k: int | None, chars: int | None) -> None:
    lines = lines_of(RENDERED[len(ATTENTION_PROMPT_AUTHORITY_INTRO):])
    wm_block = "\n".join(lines[: lines.index(OTHER_HEADINGS[0])])
    lab, _ = apply_evidence_limit({"wm_evidence": wm_block}, ("wm_evidence",), top_k, chars)
    out, _ = limit_attention_text(RENDERED, top_k, chars)
    new_lines = lines_of(out[len(ATTENTION_PROMPT_AUTHORITY_INTRO):])
    assert "\n".join(new_lines[: new_lines.index(OTHER_HEADINGS[0])]) == lab["wm_evidence"]


def sections_with(payload: Any, *, trace: str = "") -> Any:
    return PromptBuilder().build_sections(
        "hello", EmotionState(), _empty_view(), None, None, attention_payload=payload,
        conversation_trace_text=trace,
    )


def _empty_view() -> WorkingMemoryView:
    return WorkingMemoryView(
        selected=(), decisions=(), projected_bytes=0, item_capacity=8, projection_max_bytes=1024, revision=0
    )


def test_limit_sections_replaces_only_the_attention_text() -> None:
    sections = sections_with(PAYLOAD, trace="Recent conversation:\n- user: hi")
    new, numbers = limit_sections(sections, 2, 25)
    assert numbers["skipped"] is None and numbers["rows_after"] == 2
    assert new.header == sections.header and new.user_text == sections.user_text
    assert [s.name for s in new.sections] == [s.name for s in sections.sections]
    for old_s, new_s in zip(sections.sections, new.sections, strict=True):
        if old_s.name == "attention":
            assert new_s.text == limit_attention_text(old_s.text, 2, 25)[0] and new_s.text != old_s.text
            assert replace(new_s, text=old_s.text) == old_s  # origin / lead / echo untouched
        else:
            assert new_s == old_s
    old_text = next(s.text for s in sections.sections if s.name == "attention")
    new_text = next(s.text for s in new.sections if s.name == "attention")
    assert new.render() == sections.render().replace(old_text, new_text)  # nothing else in the prompt moved


def test_limit_sections_none_none_returns_the_same_object() -> None:
    sections = sections_with(PAYLOAD)
    new, numbers = limit_sections(sections, None, None)
    assert new is sections and numbers["skipped"] == "limits_off"


def test_legacy_path_evidence_sections_are_not_touched() -> None:
    sections = PromptBuilder().build_sections("hello", EmotionState(), _empty_view())
    assert {"stored_episodic_evidence", "stored_semantic_evidence"} <= {s.name for s in sections.sections}
    new, numbers = limit_sections(sections, 1, 20)
    assert new is sections and numbers["skipped"] == "legacy_evidence_section"


def test_attention_section_missing_or_ambiguous_is_a_skip() -> None:
    sections = sections_with(PAYLOAD)
    only = replace(sections, sections=tuple(s for s in sections.sections if s.name != "attention"))
    assert limit_sections(only, 1, 20)[1]["skipped"] == "attention_section_missing"
    attention = next(s for s in sections.sections if s.name == "attention")
    doubled = replace(sections, sections=(*sections.sections, attention))
    new, numbers = limit_sections(doubled, 1, 20)
    assert new is doubled and numbers["skipped"] == "attention_section_ambiguous"


def test_unparseable_attention_section_is_returned_unchanged() -> None:
    sections = sections_with(PAYLOAD)
    broken = tuple(
        replace(s, text=s.text.replace(WM_HEADING, "WM:")) if s.name == "attention" else s
        for s in sections.sections
    )
    broken_sections = replace(sections, sections=broken)
    new, numbers = limit_sections(broken_sections, 1, 20)
    assert new is broken_sections and numbers["skipped"] == "wm_evidence_heading"


def test_payload_object_and_digest_are_unchanged_by_limiting() -> None:
    payload = real_payload()
    before = (payload.rendered_text, payload.rendered_digest, payload.witnessed_bytes, payload.included_candidate_ids)
    sections = sections_with(payload)
    limit_sections(sections, 1, 20)
    limit_attention_text(payload.rendered_text, 1, 20)
    after = (payload.rendered_text, payload.rendered_digest, payload.witnessed_bytes, payload.included_candidate_ids)
    assert after == before
    # the seal still verifies and equals an independently rebuilt payload of the same inputs
    assert payload.validated_copy().rendered_digest == real_payload().rendered_digest == payload.rendered_digest
