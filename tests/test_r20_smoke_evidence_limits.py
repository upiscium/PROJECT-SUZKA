"""Per-item evidence limits of the smoke (observation only): the ``wm_evidence`` row parser against the REAL
renderer's output, the K / C limits and the never-invalid-structure property, the skip paths, the order against the
whole-section byte cap, the ``:e<K>x<C>`` condition suffix, the flags and numbers-only results.  No model, no GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_attention_prompt import WM_TEXT
from test_r20_smoke_input_caps import GIB
from test_r20_smoke_restore_matrix import FakeTransformers, _matrix_args, full_texts, matrix_results, roles_of

from experiments.r20 import bare_chat_adapter as bca
from experiments.r20 import serve, smoke
from experiments.r20.bare_chat_adapter import BareChatAdapter, apply_evidence_limit
from experiments.r20.guards import GuardRefusal
from suzka.attention.common import canonical_json

HEADING = "WorkingMemory evidence:"
ELLIPSIS = bca.ELLIPSIS


def row(text: str, kind: str = "episodic") -> str:
    """One row exactly as the real renderer writes it (the renderer's own ``canonical_json``)."""

    return canonical_json({"source_kind": kind, "text": text}).decode("ascii")


def block(*texts: str) -> str:
    return "\n".join([HEADING, *(row(text) for text in texts)])


def real_texts(message: str = "m1") -> tuple[str, str, dict[str, str]]:
    system, user = full_texts(message)
    found, _ = bca.parse_sections(system, user)
    return system, user, found


def with_rows(*texts: str, message: str = "m1") -> tuple[str, str, str]:
    """(system, user, the new wm block): the real prompt with its evidence block replaced by several rows."""

    system, user, found = real_texts(message)
    new = block(*texts)
    return system, user.replace(found["wm_evidence"], new), new


def run(system: str, user: str, **limits: Any) -> tuple[FakeTransformers, dict[str, Any]]:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, bca.RESTORE_NAMES, "system", **limits)
    adapter.generate_chat(system, user)
    report = adapter.take_record_extra()
    assert report is not None and report["applied"] is True
    return inner, report


# ------------------------------------------------------------------------------------------ the real format


def test_the_real_renderer_writes_one_ascii_json_row_per_evidence_item() -> None:
    _system, _user, found = real_texts()
    lines = found["wm_evidence"].split("\n")
    assert lines[0] == HEADING and len(lines) == 2
    assert lines[1] == row(WM_TEXT)  # the real sample equals the renderer's canonical form (escapes, surrogates)
    assert all(ord(char) < 128 for char in lines[1]) and "\\u00e9" in lines[1] and "\\ud800\\udc00" in lines[1]
    assert set(json.loads(lines[1])) == {"source_kind", "text"}


def test_the_parser_reads_every_row_in_order_from_a_real_and_a_multi_row_block() -> None:
    _system, _user, found = real_texts()
    rows, reason = bca._parse_wm_rows(found["wm_evidence"])  # noqa: SLF001
    assert reason is None and [item["text"] for item in rows] == [WM_TEXT]
    texts = ["最初の質問", "second \"quoted\" \\ back", "三番目 🙂  ", "x"]
    rows, reason = bca._parse_wm_rows(block(*texts))  # noqa: SLF001
    assert reason is None and [item["text"] for item in rows] == texts
    assert bca._parse_wm_rows(HEADING) == ([], None)  # noqa: SLF001 - no rows is not a doubt


def test_the_evidence_section_is_the_only_one_named_for_limits() -> None:
    assert bca.EVIDENCE_LIMIT_SECTIONS == ("wm_evidence",)
    system, user, found = real_texts()
    assert "Stored episodic evidence:" not in user and "Stored semantic evidence:" not in user  # legacy path only
    limited, info = apply_evidence_limit(found, bca.RESTORE_NAMES, 1, 20)
    assert {name: text for name, text in limited.items() if name != "wm_evidence"} == {n: t for n, t in found.items() if n != "wm_evidence"}
    assert info["evidence_sections_affected"] == 1


# ------------------------------------------------------------------------------------------ the limits


def test_top_k_keeps_the_first_rows_as_rendered_without_reranking() -> None:
    texts = ["zzz last alphabetically", "aaa", "mmm", "bbb", "ccc"]
    found = {"wm_evidence": block(*texts)}
    limited, info = apply_evidence_limit(found, ("wm_evidence",), 3, None)
    assert limited["wm_evidence"] == block(*texts[:3])
    assert (info["evidence_items_before"], info["evidence_items_after"], info["evidence_sections_affected"]) == (5, 3, 1)
    assert info["evidence_chars_before"] == len(found["wm_evidence"]) and info["evidence_chars_after"] == len(limited["wm_evidence"])
    assert found["wm_evidence"] == block(*texts)  # the input is never mutated


def test_item_max_chars_cuts_the_text_at_a_character_with_the_marker_counted() -> None:
    long = "あいうえお" * 20  # 100 characters
    limited, info = apply_evidence_limit({"wm_evidence": block(long, "short")}, ("wm_evidence",), None, 30)
    first, second = bca._parse_wm_rows(limited["wm_evidence"])[0]  # noqa: SLF001
    assert first["text"] == long[:29] + ELLIPSIS and len(first["text"]) == 30
    assert second["text"] == "short"  # a short row is untouched
    assert (info["evidence_items_before"], info["evidence_items_after"]) == (2, 2)
    exact = "x" * 30
    assert apply_evidence_limit({"wm_evidence": block(exact)}, ("wm_evidence",), None, 30)[0]["wm_evidence"] == block(exact)
    assert bca._parse_wm_rows(apply_evidence_limit({"wm_evidence": block(exact + "y")}, ("wm_evidence",), None, 30)[0]["wm_evidence"])[0][0]["text"] == "x" * 29 + ELLIPSIS  # noqa: SLF001


@pytest.mark.parametrize("limit", range(20, 41))
def test_a_cut_row_is_always_one_valid_renderer_line(limit: int) -> None:
    """Never an invalid structure: quotes, backslashes, controls, line separators, BMP and non-BMP characters."""

    texts = [
        WM_TEXT, 'quote=" back=\\ tab=\t nul=\x00 ls=  ps= ', "🙂" * 50, "日本語の長い文章です。" * 10,
        "é" * 40, "\ud800 lone surrogate " * 3, "a" * 500, "\\u00e9 literal escape text " * 5,
    ]
    found = {"wm_evidence": block(*texts)}
    limited, info = apply_evidence_limit(found, ("wm_evidence",), None, limit)
    lines = limited["wm_evidence"].split("\n")
    assert lines[0] == HEADING and len(lines) == len(texts) + 1 and info["evidence_limit_skipped"] == []
    for original, line in zip(texts, lines[1:]):
        parsed = json.loads(line)
        assert set(parsed) == {"source_kind", "text"} and parsed["source_kind"] == "episodic"
        assert line == row(parsed["text"]) and "\n" not in line and all(ord(char) < 128 for char in line)  # the renderer's form
        assert len(parsed["text"]) <= limit
        if len(original) > limit:
            assert parsed["text"] == original[:limit - 1] + ELLIPSIS
        else:
            assert parsed["text"] == original and line == row(original)


def test_k_and_c_compose_and_a_noop_limit_changes_nothing() -> None:
    texts = ["a" * 100, "b" * 100, "c" * 5]
    found = {"wm_evidence": block(*texts)}
    limited, info = apply_evidence_limit(found, ("wm_evidence",), 2, 40)
    assert limited["wm_evidence"] == block("a" * 39 + ELLIPSIS, "b" * 39 + ELLIPSIS)
    assert (info["evidence_items_before"], info["evidence_items_after"]) == (3, 2)
    same, info = apply_evidence_limit(found, ("wm_evidence",), 50, 2000)
    assert same == found and info["evidence_sections_affected"] == 0 and info["evidence_chars_before"] == info["evidence_chars_after"]


# ------------------------------------------------------------------------------------------ the skip paths


@pytest.mark.parametrize(("text", "reason"), [
    (HEADING + "\n- plain line", "wm_evidence_row_not_json"),
    (HEADING + "\n" + row("a") + "\n", "wm_evidence_row_not_json"),  # a trailing empty line
    (HEADING + "\n" + row("a") + "\nnot json", "wm_evidence_row_not_json"),
    (HEADING + '\n{"source_kind":"episodic","text":"a","extra":"b"}', "wm_evidence_row_shape"),
    (HEADING + '\n{"text":"a"}', "wm_evidence_row_shape"),
    (HEADING + '\n{"source_kind":"episodic","text":5}', "wm_evidence_row_shape"),
    (HEADING + '\n["episodic","a"]', "wm_evidence_row_shape"),
    (HEADING + '\n{"source_kind": "episodic", "text": "a"}', "wm_evidence_row_not_canonical"),  # spaces
    (HEADING + '\n{"text":"a","source_kind":"episodic"}', "wm_evidence_row_not_canonical"),  # key order
    (HEADING + '\n{"source_kind":"episodic","text":"café"}', "wm_evidence_row_not_canonical"),  # raw non-ASCII
    (HEADING + '\n{"source_kind":"episodic","text":"a","text":"b"}', "wm_evidence_row_not_canonical"),  # duplicate key
    ("Other heading:\n" + row("a"), "wm_evidence_heading"),
])
def test_any_parse_doubt_leaves_the_section_unchanged_and_names_the_reason(text: str, reason: str) -> None:
    found = {"header": "h", "wm_evidence": text}
    limited, info = apply_evidence_limit(found, ("header", "wm_evidence"), 1, 20)
    assert limited is found and limited["wm_evidence"] == text
    assert info["evidence_limit_skipped"] == [f"evidence_limit_skipped:{reason}"]
    assert info["evidence_sections_affected"] == 0 and info["evidence_items_before"] == info["evidence_items_after"] == 0
    assert info["evidence_chars_before"] == info["evidence_chars_after"] == len(text)


def test_nothing_to_do_when_the_section_is_not_restored_or_not_found() -> None:
    found = {"wm_evidence": block("a" * 100)}
    for restore, sections in ((("header",), found), (("wm_evidence",), {"header": "h"})):
        limited, info = apply_evidence_limit(sections, restore, 1, 20)
        assert limited is sections and info["evidence_limit_skipped"] == []
        assert (info["evidence_items_before"], info["evidence_chars_before"], info["evidence_sections_affected"]) == (0, 0, 0)
    assert apply_evidence_limit(found, ("wm_evidence",), None, None)[0] is found


def test_a_skipped_block_is_reported_by_the_adapter_and_sent_unchanged() -> None:
    system, user, _found = real_texts()
    odd = HEADING + "\n- not a json row"
    user = user.replace(real_texts()[2]["wm_evidence"], odd)
    inner, report = run(system, user, evidence_top_k=1, evidence_item_max_chars=20)
    assert report["evidence_limit_skipped"] == ["evidence_limit_skipped:wm_evidence_row_not_json"]
    assert report["evidence_sections_affected"] == 0 and "- not a json row" in roles_of(inner.sent[0])[0][1]


# ------------------------------------------------------------------------------------------ the adapter


def test_the_adapter_limits_the_rows_the_model_sees_and_reports_numbers_only() -> None:
    texts = [f"turn {index} " + "あ" * 80 for index in range(6)]
    system, user, new = with_rows(*texts)
    inner, report = run(system, user, evidence_top_k=3, evidence_item_max_chars=25)
    sent = roles_of(inner.sent[0])[0][1]
    kept = block(*[text[:24] + ELLIPSIS for text in texts[:3]])
    assert kept in sent and row(texts[3][:24] + ELLIPSIS) not in sent and "turn 5" not in sent
    assert (report["evidence_top_k_requested"], report["evidence_item_max_chars_requested"]) == (3, 25)
    assert (report["evidence_items_before"], report["evidence_items_after"]) == (6, 3)
    assert (report["evidence_chars_before"], report["evidence_chars_after"], report["evidence_sections_affected"]) == (len(new), len(kept), 1)
    assert report["evidence_limit_skipped"] == []
    limited_sections, _ = bca.parse_sections(system, user.replace(new, kept))
    assert report["restored_chars"] == sum(len(limited_sections[name]) for name in bca.RESTORE_NAMES)  # restored text IS the limited one
    for key, value in report.items():
        if key.startswith("evidence_"):
            assert isinstance(value, (int, list)), key  # numbers (and the skip labels), never the evidence text
            assert not (isinstance(value, list) and any(isinstance(item, str) and "turn 0" in item for item in value))


def test_only_the_asked_axis_is_applied_and_reported() -> None:
    system, user, _new = with_rows("a" * 60, "b" * 60, "c" * 60)
    _inner, report = run(system, user, evidence_top_k=2)
    assert report["evidence_top_k_requested"] == 2 and "evidence_item_max_chars_requested" not in report
    assert (report["evidence_items_after"], report["evidence_chars_after"] < report["evidence_chars_before"]) == (2, True)
    _inner, report = run(system, user, evidence_item_max_chars=20)
    assert report["evidence_items_before"] == report["evidence_items_after"] == 3 and "evidence_top_k_requested" not in report


@pytest.mark.parametrize("placement", ["system", "user", "first-user", "split-safe"])
def test_the_limits_apply_to_every_restore_placement(placement: str) -> None:
    system, user, _new = with_rows(*["w" * 90 for _ in range(4)])
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, bca.RESTORE_NAMES, placement, evidence_top_k=2, evidence_item_max_chars=30)
    adapter.generate_chat(system, user)
    report = adapter.take_record_extra()
    assert report is not None and report["evidence_items_after"] == 2
    sent = "\n".join(text for _role, text in roles_of(inner.sent[0]))
    cut = row("w" * 29 + ELLIPSIS)
    assert sent.count(cut) == 2 and "w" * 90 not in sent


def test_the_evidence_limit_runs_before_the_whole_section_byte_cap() -> None:
    system, user, _new = with_rows(*["x" * 300 for _ in range(6)])
    found, _ = bca.parse_sections(system, user)
    names = bca.RESTORE_NAMES
    # header .. wm_evidence of the UNLIMITED block do not fit in cap; header .. wm_evidence limited to one short row do
    limited_wm = len(block("x" * 39 + ELLIPSIS).encode())
    cap = sum(len(found[name].encode()) for name in names[:3]) + limited_wm
    _inner, uncapped = run(system, user, state_block_max_bytes=cap)
    assert "wm_evidence" in uncapped["sections_dropped_names"]  # without the limit b drops the whole evidence section
    _inner, report = run(system, user, evidence_top_k=1, evidence_item_max_chars=40, state_block_max_bytes=cap)
    assert "wm_evidence" in report["restored_sections"] and "wm_evidence" not in report["sections_dropped_names"]
    assert report["state_bytes_before"] < uncapped["state_bytes_before"]  # b sees the limited bytes
    # a section is still dropped WHOLE by b, after the rows were limited
    _inner, tight = run(system, user, evidence_top_k=1, evidence_item_max_chars=40, state_block_max_bytes=len(found["header"].encode()))
    assert tight["restored_sections"] == ["header"] and tight["evidence_items_after"] == 1 and tight["sections_dropped"] == 5


def test_the_window_composes_with_the_evidence_limit() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, bca.RESTORE_NAMES, "system", history_window=1, evidence_top_k=1, evidence_item_max_chars=20)
    for message in ("m1", "m2", "m3"):
        system, user, _ = with_rows("a" * 50, "b" * 50, message=message)
        adapter.generate_chat(system, user)
    report = adapter.take_record_extra()
    assert report is not None and report["history_pairs_sent"] == 1 and report["exchanges_dropped"] == 1
    assert report["evidence_items_before"] == 2 and report["evidence_items_after"] == 1 and report["evidence_sections_affected"] == 1


def test_defaults_are_unchanged_without_limits() -> None:
    system, user, _ = with_rows("a" * 300, "b" * 300)
    plain = FakeTransformers()
    adapter = BareChatAdapter(plain, bca.RESTORE_NAMES, "first-user")
    adapter.generate_chat(system, user)
    report = adapter.take_record_extra()
    assert report is not None and not any(key.startswith("evidence_") for key in report)
    huge = FakeTransformers()
    BareChatAdapter(huge, bca.RESTORE_NAMES, "first-user", evidence_top_k=50, evidence_item_max_chars=2000).generate_chat(system, user)
    assert plain.sent == huge.sent  # a limit that cuts nothing sends the identical messages


def test_the_adapter_validates_its_limits() -> None:
    for kwargs in ({"evidence_top_k": 0}, {"evidence_top_k": 51}, {"evidence_item_max_chars": 19}, {"evidence_item_max_chars": 2001},
                   {"evidence_top_k": True}, {"evidence_top_k": 2.0}):
        with pytest.raises(ValueError):
            BareChatAdapter(FakeTransformers(), bca.RESTORE_NAMES, "system", **kwargs)
    with pytest.raises(ValueError):
        BareChatAdapter(FakeTransformers(), (), "system", evidence_top_k=2)
    BareChatAdapter(FakeTransformers(), bca.RESTORE_NAMES, "system", evidence_top_k=1, evidence_item_max_chars=20)
    BareChatAdapter(FakeTransformers(), bca.RESTORE_NAMES, "system", evidence_top_k=50, evidence_item_max_chars=2000)


# ------------------------------------------------------------------------------------------ condition specs


def test_plain_and_old_specs_are_unchanged() -> None:
    assert [c.id for c in smoke.resolve_conditions("X10:w4:b2048,X0:w2,X7")] == ["X10:w4:b2048", "X0:w2", "X7"]
    assert all(c.evidence_top_k is None and c.evidence_item_max_chars is None for c in smoke.resolve_conditions("X10:w4:b2048,X0,X7"))


def test_the_evidence_suffix_parses_in_any_order_to_the_canonical_id() -> None:
    wanted = "X10:w6:e3x200:b2048"
    for spec in ("X10:w6:e3x200:b2048", "X10:b2048:e3x200:w6", "X10:e3x200:w6:b2048", "X10:e3x200:b2048:w6", "X10:w6:b2048:e3x200"):
        (condition,) = smoke.resolve_conditions(spec)
        assert condition.id == wanted and (condition.history_window, condition.state_block_max_bytes) == (6, 2048)
        assert (condition.evidence_top_k, condition.evidence_item_max_chars) == (3, 200)
        assert (condition.restore, condition.placement) == ("all", "split-safe")
    only = smoke.resolve_conditions("X4:e1x20,X8:e50x2000,X9:e5x200:w4")
    assert [c.id for c in only] == ["X4:e1x20", "X8:e50x2000", "X9:w4:e5x200"]
    assert "evidence 3 x 200" in smoke.resolve_conditions("X10:e3x200")[0].description
    smoke.check_conditions(smoke.resolve_conditions("X10:w6:e3x200,X10:w4:e3x200,X9:w6:e3x200,X8:w6:e3x200,X0:w6,X10:w6:e5x200"), 3)


@pytest.mark.parametrize("text", [
    "X10:e", "X10:e3", "X10:e3x", "X10:ex200", "X10:e3x200x1", "X10:e3X200", "X10:E3x200", "X10:e-3x200", "X10:e3x-200", "X10:e3.5x200",
    "X10:e3x200:e4x300", "X10:e3x200:", "X10:e3 x200", "X10:e３x200", "X10:e3x200:b", "X10:e3x200:w1:w2",
    "X10:e0x200", "X10:e51x200", "X10:e3x19", "X10:e3x2001", "X10:e999999x200", "X10:e3x99999999",
    "X0:e3x200", "X1:e3x200", "X2:e3x200", "X3:e3x200", "X5:e3x200", "X6:e3x200",  # bases that do not restore wm_evidence
    "X10:e3x200,X10:e3x200", "X10:w6:e3x200,X10:e3x200:w6",
])
def test_malformed_or_out_of_range_evidence_specs_are_refused(text: str) -> None:
    with pytest.raises(GuardRefusal) as caught:
        smoke.resolve_conditions(text)
    assert caught.value.code == "MATRIX_CONDITIONS_INVALID"


def test_check_conditions_refuses_a_bad_evidence_limit() -> None:
    def code(condition: smoke.Condition) -> str:
        with pytest.raises(GuardRefusal) as caught:
            smoke.check_conditions((condition,), 1)
        return caught.value.code

    assert code(smoke.Condition("A", "a", "all", evidence_top_k=0)) == "MATRIX_CONDITIONS_INVALID"
    assert code(smoke.Condition("A", "a", "all", evidence_item_max_chars=5)) == "MATRIX_CONDITIONS_INVALID"
    assert code(smoke.Condition("A", "a", "all", evidence_top_k=True)) == "MATRIX_CONDITIONS_INVALID"  # type: ignore[arg-type]
    assert code(smoke.Condition("A", "a", None, evidence_top_k=3)) == "MATRIX_CONDITIONS_INVALID"
    assert code(smoke.Condition("A", "a", "header", evidence_top_k=3)) == "MATRIX_CONDITIONS_INVALID"


# ------------------------------------------------------------------------------------------ the flags


def parse(base: Path, *extra: str) -> Any:
    return smoke.parser().parse_args(["--base", str(base), "--scripted", *extra])


def test_the_flags_are_validated(r20_base: Path) -> None:
    single = ["--bare-chat", "--record-model-calls"]
    for extra, code in (
        (["--evidence-top-k", "3", *single], "EVIDENCE_LIMIT_NEEDS_BARE_RESTORE"),
        (["--evidence-item-max-chars", "200", *single], "EVIDENCE_LIMIT_NEEDS_BARE_RESTORE"),
        (["--evidence-top-k", "3", *single, "--bare-restore", "values,emotion"], "EVIDENCE_LIMIT_NEEDS_WM_EVIDENCE"),
        (["--evidence-top-k", "0", *single, "--bare-restore", "all"], "EVIDENCE_LIMIT_OUT_OF_RANGE"),
        (["--evidence-top-k", "51", *single, "--bare-restore", "all"], "EVIDENCE_LIMIT_OUT_OF_RANGE"),
        (["--evidence-item-max-chars", "19", *single, "--bare-restore", "all"], "EVIDENCE_LIMIT_OUT_OF_RANGE"),
        (["--evidence-item-max-chars", "2001", *single, "--bare-restore", "all"], "EVIDENCE_LIMIT_OUT_OF_RANGE"),
        (["--matrix", "--record-model-calls", "--evidence-top-k", "3"], "MATRIX_EXCLUDES_EVIDENCE_TOP_K"),
        (["--matrix", "--record-model-calls", "--evidence-item-max-chars", "200"], "MATRIX_EXCLUDES_EVIDENCE_ITEM_MAX_CHARS"),
    ):
        with pytest.raises(GuardRefusal) as caught:
            smoke._check_arguments(parse(r20_base, *extra))  # noqa: SLF001
        assert caught.value.code == code, extra
    smoke._check_arguments(parse(r20_base, "--evidence-top-k", "3", "--evidence-item-max-chars", "200", *single, "--bare-restore", "all"))  # noqa: SLF001
    smoke._check_arguments(parse(r20_base, "--evidence-top-k", "1", *single, "--bare-restore", "wm_evidence"))  # noqa: SLF001
    assert list(r20_base.iterdir()) == []


def test_a_malformed_evidence_spec_is_refused_before_anything_exists(r20_base: Path) -> None:
    with pytest.raises(GuardRefusal) as caught:
        smoke._check_arguments(_matrix_args(r20_base, "--matrix-conditions", "X10:e3x200,X0:e3x200"))  # noqa: SLF001
    assert caught.value.code == "MATRIX_CONDITIONS_INVALID" and list(r20_base.iterdir()) == []


def test_the_serve_parser_takes_the_evidence_flags() -> None:
    args = serve.parse(["root", "--run-id", "r", "--bare-chat", "--evidence-top-k", "3", "--evidence-item-max-chars", "200"])
    assert (args.evidence_top_k, args.evidence_item_max_chars) == (3, 200)
    args = serve.parse(["root", "--run-id", "r"])
    assert (args.evidence_top_k, args.evidence_item_max_chars) == (None, None)


# ------------------------------------------------------------------------------------------ results and table


def test_a_scripted_matrix_with_evidence_conditions_runs_and_holds_numbers_only(
    r20_base: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    args = _matrix_args(r20_base, "--matrix-conditions", "X10:w1:e3x200,X0:w2,X7")
    code = smoke._smoke(args, attest=lambda *a, **k: pytest.fail("a scripted run needs no attestation"), gpu_runner=None)  # noqa: SLF001
    out = capsys.readouterr().out
    assert code == 0
    top = matrix_results(r20_base)
    assert top["requested"]["conditions"] == ["X10:w1:e3x200", "X0:w2", "X7"]
    limited, bare, plain = top["conditions"]
    assert (limited["evidence_top_k"], limited["evidence_item_max_chars"], limited["history_window"]) == (3, 200, 1)
    assert limited["runs"][0]["evidence_limits"] == {"evidence_top_k": 3, "evidence_item_max_chars": 200}
    assert "evidence_limits" not in bare["runs"][0] and "evidence_limits" not in plain["runs"][0]
    assert bare["evidence_top_k"] is None and plain["evidence_item_max_chars"] is None
    for item in top["conditions"]:
        assert {"mean_evidence_items_kept", "mean_evidence_items_dropped", "mean_evidence_chars_effective"} <= set(item)
    header = next(line for line in out.splitlines() if line.startswith("cond"))
    assert "ev kept/dropped" in header and "ev chars" in header


def test_the_default_table_has_no_evidence_columns() -> None:
    summary = smoke._condition_summary(smoke.resolve_conditions("X10:w2")[0], [], 1)  # noqa: SLF001
    header = smoke.matrix_table([summary])[0]
    assert "ev kept" not in header and "ev chars" not in header and header.split()[0] == "cond"


def test_the_condition_summary_has_mean_items_kept_dropped_and_effective_chars() -> None:
    def turn(before: int, after: int, chars_before: int, chars_after: int, affected: int, skipped: list[str]) -> dict[str, Any]:
        return {"bare_chat": {
            "evidence_items_before": before, "evidence_items_after": after, "evidence_chars_before": chars_before,
            "evidence_chars_after": chars_after, "evidence_sections_affected": affected, "evidence_limit_skipped": skipped,
        }, "label": "E", "ordinal": 1, "seconds": 1.0, "response_chars": 3, "option_naming": None,
            "observed_labels": None, "prompt_assembly": None, "chat_roles": None}

    run_row = {"ended": "completed", "gpu": {"min_free": 2 * GIB, "peak_allocated": GIB},
               "turns": [turn(6, 3, 900, 500, 1, []), turn(2, 2, 300, 300, 0, []), turn(0, 0, 40, 40, 0, ["evidence_limit_skipped:x"])]}
    condition = smoke.resolve_conditions("X10:w6:e3x200")[0]
    summary = smoke._condition_summary(condition, [run_row], 1)  # noqa: SLF001
    assert summary["mean_evidence_items_before"] == round(8 / 3, 1) and summary["mean_evidence_items_kept"] == round(5 / 3, 1)
    assert summary["mean_evidence_items_dropped"] == 1.0
    assert summary["mean_evidence_chars_before"] == round(1240 / 3, 1) and summary["mean_evidence_chars_effective"] == round(840 / 3, 1)
    assert summary["mean_evidence_sections_affected"] == round(1 / 3, 1) and summary["evidence_limit_skipped_turns"] == 1
    table = smoke.matrix_table([summary])
    assert "ev kept/dropped" in table[0] and table[1].split()[-2:] == ["1.7/1.0", "280"]
    assert all(isinstance(value, (int, float, str, type(None), list, dict)) for value in summary.values())
