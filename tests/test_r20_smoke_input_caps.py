"""Lab VRAM measurement and input caps of the smoke (observation only): the sampler record shape (fake torch),
``--history-window`` / ``--state-block-max-bytes`` in the bare-chat adapter, the matrix condition-spec parsing, a
scripted matrix with suffixed conditions and numbers-only results.  No model, no GPU."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest
from test_r20_smoke_restore_matrix import FakeTransformers, _matrix_args, full_texts, matrix_results, roles_of

from experiments.r20 import bare_chat_adapter as bca
from experiments.r20 import measure, serve, smoke
from experiments.r20.bare_chat_adapter import BARE_SYSTEM_TEXT, BareChatAdapter, apply_state_cap
from experiments.r20.guards import GuardRefusal
from experiments.r20.recording_provider import RecordingProvider, read_calls

GIB = 1024**3


# ------------------------------------------------------------------------------------------ the sampler


def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = types.SimpleNamespace(
        mem_get_info=lambda: (5 * GIB, 24 * GIB), memory_reserved=lambda: 8 * GIB,
        memory_allocated=lambda: 6 * GIB, max_memory_allocated=lambda: 7 * GIB,
        is_available=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_the_torch_reader_carries_allocated_and_peak_allocated(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch)
    assert measure.torch_reader() == (5 * GIB, 24 * GIB, 8 * GIB, 6 * GIB, 7 * GIB)


def test_a_sample_row_has_allocated_and_max_allocated_next_to_reserved_and_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_torch(monkeypatch)
    sampler = measure.GpuSampler(tmp_path / "gpu.jsonl")
    sampler.sample("tick")
    (row,) = measure.read_log(tmp_path / "gpu.jsonl")
    assert {"free", "total", "reserved", "allocated", "max_allocated", "t", "pid", "label"} == set(row)
    assert (row["free"], row["reserved"], row["allocated"], row["max_allocated"]) == (5 * GIB, 8 * GIB, 6 * GIB, 7 * GIB)
    assert all(isinstance(value, (int, float)) or key == "label" for key, value in row.items())  # numbers (and the label)


def test_a_three_value_reader_still_works_and_the_summary_takes_the_peak(tmp_path: Path) -> None:
    sampler = measure.GpuSampler(tmp_path / "gpu.jsonl", reader=lambda: (1, 2, 3))
    sampler.sample("tick")
    (row,) = measure.read_log(tmp_path / "gpu.jsonl")
    assert "allocated" not in row and "max_allocated" not in row
    assert "peak_allocated" not in measure.summarize([row])
    summary = measure.summarize([
        {"free": 9, "total": 20, "reserved": 5, "allocated": 4, "max_allocated": 6},
        {"free": 3, "total": 20, "reserved": 7, "allocated": 4, "max_allocated": 11},
        {"error": "RuntimeError"},
    ])
    assert summary["peak_allocated"] == 11 and summary["min_free"] == 3 and summary["peak_reserved"] == 7


# ------------------------------------------------------------------------------------------ the state cap


def test_apply_state_cap_drops_whole_sections_from_the_end() -> None:
    system, user = full_texts("m1")
    found, _ = bca.parse_sections(system, user)
    names = bca.RESTORE_NAMES
    total = sum(len(found[name].encode("utf-8")) for name in names)
    assert apply_state_cap(names, found, None) == (names, [], total, total)
    assert apply_state_cap(names, found, total)[1] == []
    kept, dropped, before, after = apply_state_cap(names, found, total - 1)
    assert kept == names[:-1] and dropped == ["emotion"] and before == total and after == total - len(found["emotion"].encode())
    kept, dropped, _, after = apply_state_cap(names, found, 0)
    assert kept == () and dropped == list(names) and after == 0
    first = len(found["header"].encode("utf-8"))
    assert apply_state_cap(names, found, first)[0] == ("header",)  # a section is never cut: header alone fits exactly
    assert apply_state_cap(names, found, first - 1)[0] == ()


def test_the_utf8_bytes_not_the_characters_are_counted() -> None:
    found = {"header": "あ" * 10}  # 10 chars, 30 bytes
    assert apply_state_cap(("header",), found, 29)[1] == ["header"]
    assert apply_state_cap(("header",), found, 30)[1] == []


@pytest.mark.parametrize("placement", ["system", "user", "first-user", "split-safe"])
def test_the_state_cap_applies_to_every_restore_placement(placement: str) -> None:
    system, user = full_texts("m1")
    found, _ = bca.parse_sections(system, user)
    cap = len(found["header"].encode()) + len(found["context"].encode())
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, bca.RESTORE_NAMES, placement, state_block_max_bytes=cap)
    adapter.generate_chat(system, user)
    report = adapter.take_record_extra()
    assert report is not None and report["applied"] is True
    assert report["state_block_max_bytes_requested"] == cap
    assert report["sections_dropped"] == 4 and report["sections_dropped_names"] == ["values", "wm_evidence", "motivations_goals", "emotion"]
    assert report["state_bytes_effective"] == cap <= report["state_bytes_before"] and report["restored_sections"] == ["header", "context"]
    sent = json.dumps(inner.sent[0], ensure_ascii=False)
    assert found["header"][:40] in sent and "Active Values:" not in sent and "Emotion:" not in sent
    assert roles_of(inner.sent[0])[-1][1].endswith("m1")


def test_a_zero_state_cap_is_the_bare_chat_and_no_cap_is_unchanged() -> None:
    system, user = full_texts("m1")
    inner = FakeTransformers()
    BareChatAdapter(inner, bca.RESTORE_NAMES, "system", state_block_max_bytes=0).generate_chat(system, user)
    assert roles_of(inner.sent[0]) == [("system", BARE_SYSTEM_TEXT), ("user", "m1")]
    plain = FakeTransformers()
    capped = FakeTransformers()
    BareChatAdapter(plain, bca.RESTORE_NAMES, "first-user").generate_chat(system, user)
    BareChatAdapter(capped, bca.RESTORE_NAMES, "first-user", state_block_max_bytes=10**7).generate_chat(system, user)
    assert plain.sent == capped.sent


def test_without_caps_the_report_has_no_cap_keys() -> None:
    adapter = BareChatAdapter(FakeTransformers(), bca.RESTORE_NAMES, "system")
    adapter.generate_chat(*full_texts("m1"))
    report = adapter.take_record_extra()
    assert report is not None
    assert not any(key.startswith(("history_window", "state_block", "state_bytes", "sections_dropped", "exchanges_dropped")) for key in report)


# ------------------------------------------------------------------------------------------ the history window


def test_the_history_window_keeps_the_most_recent_exchanges_and_never_truncates_text() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, history_window=2)
    long_user = "長い" * 500
    for message in ("m1", "m2", long_user, "m4"):
        adapter.generate_chat(*full_texts(message))
    fourth = roles_of(inner.sent[3])
    assert fourth == [("system", BARE_SYSTEM_TEXT), ("user", "m2"), ("assistant", "out2"), ("user", long_user), ("assistant", "out3"), ("user", "m4")]
    assert [text for role, text in roles_of(inner.sent[2]) if role == "user"] == ["m1", "m2", long_user]
    report = adapter.take_record_extra()
    assert report is not None and report["history_window_requested"] == 2
    assert report["history_pairs"] == 3 and report["history_pairs_sent"] == 2 and report["exchanges_dropped"] == 1
    assert len(adapter._history) == 4  # noqa: SLF001  the stored history stays whole


def test_the_stored_history_stays_whole_under_a_window() -> None:
    adapter = BareChatAdapter(FakeTransformers(), history_window=1)
    for message in ("m1", "m2", "m3"):
        adapter.generate_chat(*full_texts(message))
    assert adapter._history == [("m1", "out1"), ("m2", "out2"), ("m3", "out3")]  # noqa: SLF001


def test_a_zero_window_sends_no_history_and_a_large_window_all_of_it() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, history_window=0)
    for message in ("m1", "m2", "m3"):
        adapter.generate_chat(*full_texts(message))
    assert roles_of(inner.sent[2]) == [("system", BARE_SYSTEM_TEXT), ("user", "m3")]
    report = adapter.take_record_extra()
    assert report is not None and report["history_pairs_sent"] == 0 and report["exchanges_dropped"] == 2
    wide = FakeTransformers()
    other = BareChatAdapter(wide, history_window=99)
    for message in ("m1", "m2", "m3"):
        other.generate_chat(*full_texts(message))
    assert [r for r, _t in roles_of(wide.sent[2])] == ["system", "user", "assistant", "user", "assistant", "user"]
    report = other.take_record_extra()
    assert report is not None and report["exchanges_dropped"] == 0


def test_the_first_user_block_moves_to_the_first_kept_exchange_under_a_window() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, ("emotion",), "first-user", history_window=1)
    for message in ("m1", "m2", "m3"):
        adapter.generate_chat(*full_texts(message))
    roles = roles_of(inner.sent[2])
    assert [r for r, _t in roles] == ["system", "user", "assistant", "user"]
    assert "Emotion:" in roles[1][1] and roles[1][1].endswith("m2") and roles[-1] == ("user", "m3")
    assert roles[2] == ("assistant", "out2")  # assistant text raw


def test_window_and_cap_without_restore_are_plain_and_both_are_validated() -> None:
    inner = FakeTransformers()
    BareChatAdapter(inner, history_window=3, state_block_max_bytes=5).generate_chat(*full_texts("m1"))
    assert roles_of(inner.sent[0]) == [("system", BARE_SYSTEM_TEXT), ("user", "m1")]
    for bad in (-1, True, 1.5, "3"):
        with pytest.raises(ValueError):
            BareChatAdapter(FakeTransformers(), history_window=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            BareChatAdapter(FakeTransformers(), state_block_max_bytes=bad)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------ tokens and peak hooks


class _Ids:
    shape = (1, 321)


class TokenInner(FakeTransformers):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def _chat_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"messages": messages, "input_ids": _Ids()}

    def _generate_from(self, inputs: dict[str, Any]) -> str:
        self.events.append("generate")
        return super()._generate_from(inputs)


def test_the_report_has_the_input_token_count_and_the_per_call_peak() -> None:
    events: list[str] = []
    adapter = BareChatAdapter(
        TokenInner(events), reset_peak=lambda: events.append("reset"),
        read_peak=lambda: (events.append("read") or (7 * GIB, 9 * GIB)),
    )
    adapter.generate_chat(*full_texts("m1"))
    report = adapter.take_record_extra()
    assert events == ["reset", "generate", "read"]  # the peak is reset right before the model call
    assert report is not None and report["input_tokens"] == 321
    assert report["peak_allocated_bytes"] == 7 * GIB and report["reserved_bytes"] == 9 * GIB


def test_unreadable_tokens_and_failing_hooks_are_gaps_not_crashes() -> None:
    def boom() -> Any:
        raise RuntimeError("no cuda")

    adapter = BareChatAdapter(FakeTransformers(), reset_peak=boom, read_peak=boom)
    assert adapter.generate_chat(*full_texts("m1")) == "out1"
    report = adapter.take_record_extra()
    assert report is not None and report["input_tokens"] is None
    assert "peak_allocated_bytes" not in report and "reserved_bytes" not in report


def test_the_recorded_numbers_hold_no_text(tmp_path: Path) -> None:
    inner = TokenInner([])
    adapter = BareChatAdapter(
        inner, bca.RESTORE_NAMES, "split-safe", history_window=1, state_block_max_bytes=300,
        reset_peak=lambda: None, read_peak=lambda: (1, 2),
    )
    recorder = RecordingProvider(adapter, tmp_path)
    for message in ("次の質問です", "その次"):
        system, user = full_texts(message)
        recorder.generate_chat(system, user)
    row = read_calls(tmp_path)[-1]
    shown = bca.numbers(row)
    assert shown is not None and "messages_sent" not in shown
    found, _ = bca.parse_sections(system, user)
    dumped = json.dumps(shown, ensure_ascii=False)
    assert not any(text in dumped for text in found.values()) and "その次" not in dumped and "out1" not in dumped
    for key, value in shown.items():  # numbers, booleans, None, and labels only
        assert isinstance(value, (int, float, bool, str, type(None), list, dict)), key
        if isinstance(value, list):
            assert all(isinstance(item, str) for item in value), key


# ------------------------------------------------------------------------------------------ condition specs


def _ids(text: str) -> list[str]:
    return [condition.id for condition in smoke.resolve_conditions(text)]


def test_the_default_conditions_and_plain_ids_are_unchanged() -> None:
    assert smoke.resolve_conditions(None) is smoke.CONDITIONS
    plain = smoke.resolve_conditions("X0,X7,X9,X10")
    assert [c.id for c in plain] == ["X0", "X7", "X9", "X10"]
    assert all(c.history_window is None and c.state_block_max_bytes is None for c in plain)
    assert plain[1] is smoke.CONDITIONS[7]


def test_suffixed_specs_parse_in_either_order_to_the_canonical_id() -> None:
    conditions = smoke.resolve_conditions("X10:w6:b4096,X10:b2048:w4,X9:w4,X8:b0,X0:w4, X7:w0")
    assert [c.id for c in conditions] == ["X10:w6:b4096", "X10:w4:b2048", "X9:w4", "X8:b0", "X0:w4", "X7:w0"]
    first, second, third, fourth, fifth, sixth = conditions
    assert (first.history_window, first.state_block_max_bytes, first.restore, first.placement) == (6, 4096, "all", "split-safe")
    assert (second.history_window, second.state_block_max_bytes) == (4, 2048)
    assert (third.history_window, third.state_block_max_bytes, third.placement) == (4, None, "first-user")
    assert (fourth.history_window, fourth.state_block_max_bytes) == (None, 0)
    assert (fifth.history_window, fifth.restore) == (4, None)
    assert sixth.history_window == 0
    assert "window 6" in first.description and "4096" in first.description
    smoke.check_conditions(conditions, 5)


@pytest.mark.parametrize("text", [
    "X10:w", "X10:b", "X10:w-1", "X10:w4.5", "X10:w+4", "X10:x4", "X10:w4:w2", "X10:b1:b2", "X10:w4:", ":w4", "X10::w4",
    "X0:b100", "X99:w4", "x10:w4", "X10:W4", "X10:w４", "X10:w4:b", "X10:w4:b2048:w1", "X10 :w4:b1 x", "X10:w4:c1",
    "X10:w99999999999", "X10:b10000001", "X10,,X7", "", "X10:w4,X10:w4", "X10:w4:b2,X10:b2:w4", "X10,X10",
])
def test_malformed_specs_are_refused(text: str) -> None:
    with pytest.raises(GuardRefusal) as caught:
        smoke.resolve_conditions(text)
    assert caught.value.code == "MATRIX_CONDITIONS_INVALID"


def test_the_same_base_with_different_caps_is_not_a_duplicate() -> None:
    assert _ids("X10:w4,X10:w2,X10") == ["X10:w4", "X10:w2", "X10"]


def test_check_conditions_refuses_a_bad_cap_and_keeps_the_60_subject_cap() -> None:
    def code(conditions: tuple[smoke.Condition, ...], repeats: int) -> str:
        with pytest.raises(GuardRefusal) as caught:
            smoke.check_conditions(conditions, repeats)
        return caught.value.code

    assert code((smoke.Condition("A", "a", None, history_window=-1),), 1) == "MATRIX_CONDITIONS_INVALID"
    assert code((smoke.Condition("A", "a", "all", "system", state_block_max_bytes=True),), 1) == "MATRIX_CONDITIONS_INVALID"  # type: ignore[arg-type]
    assert code((smoke.Condition("A", "a", None, state_block_max_bytes=5),), 1) == "MATRIX_CONDITIONS_INVALID"
    suffixed = smoke.resolve_conditions(",".join(f"X10:w{index}:b2048" for index in range(13)))
    assert code(suffixed, 5) == "MATRIX_TOO_MANY_SUBJECTS"  # 65 > 60
    smoke.check_conditions(suffixed[:12], 5)


# ------------------------------------------------------------------------------------------ the flags


def parse(base: Path, *extra: str) -> Any:
    return smoke.parser().parse_args(["--base", str(base), "--scripted", *extra])


def test_the_global_cap_flags_are_validated(r20_base: Path) -> None:
    for extra, code in (
        (["--history-window", "2"], "INPUT_CAPS_NEED_BARE_CHAT"),
        (["--state-block-max-bytes", "9", "--bare-chat", "--record-model-calls"], "STATE_CAP_NEEDS_BARE_RESTORE"),
        (["--history-window", "-1", "--bare-chat", "--record-model-calls"], "INPUT_CAP_OUT_OF_RANGE"),
        (["--state-block-max-bytes", "-5", "--bare-chat", "--record-model-calls", "--bare-restore", "all"], "INPUT_CAP_OUT_OF_RANGE"),
        (["--history-window", str(smoke.MAX_CAP + 1), "--bare-chat", "--record-model-calls"], "INPUT_CAP_OUT_OF_RANGE"),
        (["--matrix", "--record-model-calls", "--history-window", "2"], "MATRIX_EXCLUDES_HISTORY_WINDOW"),
        (["--matrix", "--record-model-calls", "--history-window", "0"], "MATRIX_EXCLUDES_HISTORY_WINDOW"),
        (["--matrix", "--record-model-calls", "--state-block-max-bytes", "0"], "MATRIX_EXCLUDES_STATE_BLOCK_MAX_BYTES"),
    ):
        with pytest.raises(GuardRefusal) as caught:
            smoke._check_arguments(parse(r20_base, *extra))  # noqa: SLF001
        assert caught.value.code == code, extra
    smoke._check_arguments(parse(r20_base, "--history-window", "0", "--bare-chat", "--record-model-calls"))  # noqa: SLF001
    smoke._check_arguments(parse(  # noqa: SLF001
        r20_base, "--history-window", "4", "--state-block-max-bytes", "0", "--bare-chat", "--record-model-calls",
        "--bare-restore", "all",
    ))
    assert list(r20_base.iterdir()) == []


def test_a_malformed_matrix_spec_is_refused_before_anything_exists(r20_base: Path) -> None:
    args = _matrix_args(r20_base, "--matrix-conditions", "X10:w4,X0:b9")
    with pytest.raises(GuardRefusal) as caught:
        smoke._check_arguments(args)  # noqa: SLF001
    assert caught.value.code == "MATRIX_CONDITIONS_INVALID" and list(r20_base.iterdir()) == []


def test_the_serve_parser_takes_the_cap_flags() -> None:
    args = serve.parse(["root", "--run-id", "r", "--bare-chat", "--history-window", "4", "--state-block-max-bytes", "2048"])
    assert (args.history_window, args.state_block_max_bytes) == (4, 2048)
    args = serve.parse(["root", "--run-id", "r"])
    assert (args.history_window, args.state_block_max_bytes) == (None, None)


# ------------------------------------------------------------------------------------------ the scripted matrix


def test_a_scripted_matrix_with_suffixed_conditions_runs_and_holds_numbers_only(
    r20_base: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    args = _matrix_args(r20_base, "--matrix-conditions", "X7:w1:b64,X0:w2,X10")
    code = smoke._smoke(args, attest=lambda *a, **k: pytest.fail("a scripted run needs no attestation"), gpu_runner=None)  # noqa: SLF001
    out = capsys.readouterr().out
    assert code == 0
    top = matrix_results(r20_base)
    assert top["requested"]["conditions"] == ["X7:w1:b64", "X0:w2", "X10"] and top["requested"]["subjects_planned"] == 3
    assert top["completed_conditions"] == ["X7:w1:b64", "X0:w2", "X10"]
    capped, window_only, plain = top["conditions"]
    assert (capped["history_window"], capped["state_block_max_bytes"], capped["restore"], capped["placement"]) == (1, 64, "all", "system")
    assert (window_only["history_window"], window_only["state_block_max_bytes"], window_only["restore"]) == (2, None, None)
    assert plain["history_window"] is None and plain["state_block_max_bytes"] is None
    assert capped["runs"][0]["caps"] == {"history_window": 1, "state_block_max_bytes": 64}
    assert window_only["runs"][0]["caps"] == {"history_window": 2, "state_block_max_bytes": None}
    assert "caps" not in plain["runs"][0]
    for item in top["conditions"]:
        assert {"max_input_tokens", "peak_allocated_bytes", "min_free_bytes", "max_reserved_bytes"} <= set(item)
    dumped = json.dumps(top, ensure_ascii=False)
    assert "synthetic fixture answer" not in dumped and "Emotion:" not in dumped
    header = next(line for line in out.splitlines() if line.startswith("cond"))
    assert "min free GiB" in header and "peak alloc GiB" in header and "max input tokens" in header
    assert "X7:w1:b64" in out and "completed conditions: X7:w1:b64, X0:w2, X10" in out


def test_the_condition_summary_surfaces_tokens_peak_and_effective_caps() -> None:
    def turn(tokens: int, peak: int, dropped: int, sections: int, state: int) -> dict[str, Any]:
        return {"bare_chat": {
            "input_tokens": tokens, "peak_allocated_bytes": peak, "reserved_bytes": peak + GIB,
            "exchanges_dropped": dropped, "sections_dropped": sections, "state_bytes_effective": state,
        }, "label": "E", "ordinal": 1, "seconds": 1.0, "response_chars": 3, "option_naming": None,
            "observed_labels": None, "prompt_assembly": None, "chat_roles": None}

    run = {"ended": "completed", "gpu": {"min_free": 2 * GIB, "peak_allocated": 10 * GIB},
           "turns": [turn(100, 6 * GIB, 0, 0, 100), turn(900, 8 * GIB, 2, 3, 40)]}
    condition = smoke.resolve_conditions("X10:w2:b100")[0]
    summary = smoke._condition_summary(condition, [run], 1)  # noqa: SLF001
    assert summary["max_input_tokens"] == 900 and summary["peak_allocated_bytes"] == 10 * GIB
    assert summary["min_free_bytes"] == 2 * GIB and summary["max_reserved_bytes"] == 9 * GIB
    assert summary["mean_exchanges_dropped"] == 1.0 and summary["mean_sections_dropped"] == 1.5
    assert summary["history_window"] == 2 and summary["state_block_max_bytes"] == 100
    row = smoke.matrix_table([summary])[1].split()
    assert row[0] == "X10:w2:b100" and "10.0" in row and "2.0" in row and "900" in row
