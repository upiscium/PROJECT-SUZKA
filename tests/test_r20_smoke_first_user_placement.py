"""``--restore-placement first-user|split-safe`` and ``--matrix-conditions`` (X9, X10) of the smoke (observation only).
Message construction of the #414-compliant placements, flag / condition validation and the results (numbers only).
No model, no GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_r20_smoke_restore_matrix import FakeTransformers, full_texts, matrix_results, parse, roles_of

from experiments.r20 import bare_chat_adapter as bca
from experiments.r20 import smoke
from experiments.r20.bare_chat_adapter import BARE_SYSTEM_TEXT, BareChatAdapter
from experiments.r20.guards import GuardRefusal
from suzka.persona.prompt_builder import RUNTIME_DATA_HEADING

MARK = "\n\nUser message:\n"
STORED = ("values", "wm_evidence", "motivations_goals")
SAFE = ("header", "context", "emotion")
DATA_ORDER = ("context", "values", "wm_evidence", "motivations_goals", "emotion")


def _block(found: dict[str, str], names: tuple[str, ...], *, header: bool) -> str:
    head = [found["header"]] if header else []
    data = "\n\n".join(found[name] for name in names)
    return "\n\n".join([*head, RUNTIME_DATA_HEADING + "\n" + data])


def test_placements_and_the_split_constant() -> None:
    assert bca.PLACEMENTS == ("system", "user", "first-user", "split-safe")
    assert bca.SYSTEM_SAFE_SECTIONS == frozenset(SAFE)
    assert set(bca.RESTORE_NAMES) == set(SAFE) | set(STORED)


def test_first_user_first_call_puts_the_block_in_the_only_user_message() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, bca.RESTORE_NAMES, "first-user")
    system, user = full_texts("m1")
    found, _ = bca.parse_sections(system, user)
    adapter.generate_chat(system, user)
    (sent,) = inner.sent
    expected = _block(found, DATA_ORDER, header=True) + MARK + "m1"
    assert roles_of(sent) == [("system", BARE_SYSTEM_TEXT), ("user", expected)]
    report = adapter.take_record_extra()
    assert report is not None and report["restore_placement"] == "first-user"
    assert report["restored_chars_system"] == 0
    assert report["restored_chars_first_user"] == report["restored_chars"] == sum(len(t) for t in found.values())


def test_first_user_reattaches_the_current_data_to_the_first_turn_and_keeps_history_raw() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, bca.RESTORE_NAMES, "first-user")
    founds = []
    for message in ("m1", "m2", "m3"):
        system, user = full_texts(message)
        user = user.replace("- valence: 0.0", f"- valence: 0.{len(founds) + 1}")  # state changes per call
        founds.append(bca.parse_sections(system, user)[0])
        adapter.generate_chat(system, user)
    assert len({found["emotion"] for found in founds}) == 3
    for index, sent in enumerate(inner.sent):
        roles = roles_of(sent)
        expected_first = _block(founds[index], DATA_ORDER, header=True)  # the CURRENT call's data
        if index == 0:
            assert roles[-1] == ("user", expected_first + MARK + "m1")
            continue
        assert roles[1] == ("user", expected_first + MARK + "m1")
        assert roles[-1] == ("user", f"m{index + 1}")  # final user message: the user's text only
        assert [r for r, _t in roles] == ["system", *["user", "assistant"] * index, "user"]
    third = roles_of(inner.sent[2])
    assert third[2:5] == [("assistant", "out1"), ("user", "m2"), ("assistant", "out2")]  # later pairs untouched
    assert adapter._history == [("m1", "out1"), ("m2", "out2"), ("m3", "out3")]  # noqa: SLF001 - never persisted


def test_split_safe_sends_code_generated_sections_to_system_and_stored_text_to_the_first_user_turn() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner, bca.RESTORE_NAMES, "split-safe")
    system, user = full_texts("m1")
    found, _ = bca.parse_sections(system, user)
    adapter.generate_chat(system, user)
    report = adapter.take_record_extra()
    adapter.generate_chat(*full_texts("m2"))
    first, second = (roles_of(sent) for sent in inner.sent)
    expected_system = BARE_SYSTEM_TEXT + "".join("\n\n" + found[name] for name in bca.RESTORE_NAMES if name in SAFE)
    block = RUNTIME_DATA_HEADING + "\n" + "\n\n".join(found[name] for name in bca.RESTORE_NAMES if name in STORED)
    assert first == [("system", expected_system), ("user", block + MARK + "m1")]
    assert second[0] == ("system", expected_system)
    assert second[1] == ("user", block + MARK + "m1") and second[3] == ("user", "m2")
    for name in STORED:  # nothing that can carry stored / model text is in the system role
        assert found[name] not in expected_system
    for name in SAFE:
        assert found[name] not in block
    assert report is not None
    assert report["restored_chars_system"] == sum(len(found[n]) for n in SAFE)
    assert report["restored_chars_first_user"] == sum(len(found[n]) for n in STORED)


def test_split_safe_without_stored_sections_has_no_block_and_a_header_only_block_has_no_heading() -> None:
    inner = FakeTransformers()
    BareChatAdapter(inner, ("emotion", "context"), "split-safe").generate_chat(*full_texts("m1"))
    roles = roles_of(inner.sent[0])
    assert roles[1] == ("user", "m1") and "Emotion:" in roles[0][1]
    inner = FakeTransformers()
    BareChatAdapter(inner, ("header",), "first-user").generate_chat(*full_texts("m1"))
    roles = roles_of(inner.sent[0])
    assert roles[1][1].endswith(MARK + "m1") and RUNTIME_DATA_HEADING not in roles[1][1] and roles[0][1] == BARE_SYSTEM_TEXT


def test_the_other_placements_and_the_flag_off_report_are_unchanged() -> None:
    adapter = BareChatAdapter(FakeTransformers(), bca.RESTORE_NAMES, "user")
    adapter.generate_chat(*full_texts("m1"))
    report = adapter.take_record_extra()
    assert report is not None and "restored_chars_system" not in report and "restored_chars_first_user" not in report
    plain = BareChatAdapter(FakeTransformers())
    plain.generate_chat(*full_texts("m1"))
    assert not any(key.startswith("restore") for key in (plain.take_record_extra() or {}))


def test_a_skipped_call_has_zero_destinations_and_no_history() -> None:
    adapter = BareChatAdapter(FakeTransformers(), ("emotion",), "split-safe")
    adapter.generate_chat("SYS", "garbled")
    report = adapter.take_record_extra()
    assert report is not None and report["restored_chars_system"] == 0 and report["restored_chars_first_user"] == 0
    assert adapter._history == []  # noqa: SLF001


def test_the_report_holds_numbers_and_labels_only() -> None:
    adapter = BareChatAdapter(FakeTransformers(), bca.RESTORE_NAMES, "split-safe")
    system, user = full_texts("m1")
    adapter.generate_chat(system, user)
    shown = bca.numbers({bca.RECORD_KEY: adapter.take_record_extra()})
    assert shown is not None
    dumped = json.dumps(shown, ensure_ascii=False)
    found, _ = bca.parse_sections(system, user)
    assert not any(text in dumped for text in found.values()) and "out1" not in dumped and "messages_sent" not in shown


# ------------------------------------------------------------------------------------------ flags and conditions


def test_placement_flag_validation(r20_base: Path) -> None:
    for placement in ("first-user", "split-safe"):
        with pytest.raises(GuardRefusal) as caught:
            smoke._check_arguments(parse(r20_base, "--bare-chat", "--record-model-calls", "--restore-placement", placement))
        assert caught.value.code == "RESTORE_PLACEMENT_NEEDS_BARE_RESTORE"
        with pytest.raises(GuardRefusal) as caught:
            smoke._check_arguments(parse(r20_base, "--matrix", "--record-model-calls", "--restore-placement", placement))
        assert caught.value.code == "MATRIX_EXCLUDES_RESTORE_PLACEMENT"
        smoke._check_arguments(parse(
            r20_base, "--bare-chat", "--record-model-calls", "--bare-restore", "all", "--restore-placement", placement,
        ))
    with pytest.raises(SystemExit):
        parse(r20_base, "--restore-placement", "middle")
    assert list(r20_base.iterdir()) == []


def test_extra_conditions_and_the_default_list() -> None:
    assert [(c.id, c.restore, c.placement) for c in smoke.EXTRA_CONDITIONS] == [
        ("X9", "all", "first-user"), ("X10", "all", "split-safe"),
    ]
    assert [c.id for c in smoke.CONDITIONS] == [f"X{i}" for i in range(9)]  # default unchanged
    assert smoke.resolve_conditions(None) is smoke.CONDITIONS
    assert [c.id for c in smoke.resolve_conditions("X0, X7,X9,X10")] == ["X0", "X7", "X9", "X10"]
    assert [c.id for c in smoke.resolve_conditions("X10,X0")] == ["X10", "X0"]
    smoke.check_conditions(smoke.resolve_conditions("X0,X7,X8,X9,X10"), 5)
    for bad in ("", "X0,", "X0,X0", "X11", "x9", "X9;X10"):
        with pytest.raises(GuardRefusal) as caught:
            smoke.resolve_conditions(bad)
        assert caught.value.code == "MATRIX_CONDITIONS_INVALID", bad


def test_matrix_conditions_argument_validation(r20_base: Path) -> None:
    for extra, code in (
        (["--record-model-calls", "--matrix-conditions", "X9"], "MATRIX_CONDITIONS_NEEDS_MATRIX"),
        (["--matrix", "--record-model-calls", "--matrix-conditions", "X99"], "MATRIX_CONDITIONS_INVALID"),
        (["--matrix", "--record-model-calls", "--matrix-conditions", "X9,X9"], "MATRIX_CONDITIONS_INVALID"),
        (["--matrix", "--record-model-calls", "--matrix-conditions", ""], "MATRIX_CONDITIONS_INVALID"),
    ):
        with pytest.raises(GuardRefusal) as caught:
            smoke._check_arguments(parse(r20_base, *extra))
        assert caught.value.code == code, extra
        assert smoke.main(["--base", str(r20_base), "--scripted", *extra]) == smoke.EXIT_REFUSED
    smoke._check_arguments(parse(r20_base, "--matrix", "--record-model-calls", "--matrix-conditions", "X0,X7,X9,X10"))
    assert list(r20_base.iterdir()) == []


def test_all_eleven_conditions_fit_the_cap(r20_base: Path) -> None:
    ids = ",".join(f"X{i}" for i in range(11))  # 11 x 5 = 55 <= 60
    smoke._check_arguments(parse(r20_base, "--matrix", "--record-model-calls", "--matrix-conditions", ids))
    with pytest.raises(GuardRefusal) as caught:
        smoke.check_conditions(smoke.resolve_conditions(ids), 6)  # 66 > 60
    assert caught.value.code == "MATRIX_TOO_MANY_SUBJECTS"


def test_a_selected_matrix_runs_end_to_end_with_destination_numbers_and_no_text(
    r20_base: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    args = smoke.parser().parse_args([
        "--base", str(r20_base), "--scripted", "--matrix", "--record-model-calls", "--turns", "2", "--repeats", "1",
        "--matrix-conditions", "X7,X9,X10",
    ])
    code = smoke._smoke(args, attest=lambda *a, **k: pytest.fail("no attestation"), gpu_runner=None)  # noqa: SLF001
    capsys.readouterr()
    top = matrix_results(r20_base)
    assert code == 0 and top["requested"]["conditions"] == ["X7", "X9", "X10"]
    assert top["requested"]["subjects_planned"] == 3 and top["completed_conditions"] == ["X7", "X9", "X10"]
    by_id: dict[str, Any] = {item["id"]: item for item in top["conditions"]}
    assert (by_id["X9"]["restore"], by_id["X9"]["placement"]) == ("all", "first-user")
    assert (by_id["X10"]["restore"], by_id["X10"]["placement"]) == ("all", "split-safe")
    assert "mean_restored_chars_system" in by_id["X9"] and "mean_restored_chars_system" not in by_id["X7"]
    assert by_id["X10"]["runs"][0]["restore"]["placement"] == "split-safe"
    turn = by_id["X9"]["runs"][0]["turns"][0]["bare_chat"]  # scripted: not a chat call, but the keys are there
    assert turn["restore_placement"] == "first-user" and turn["restored_chars_system"] == 0
    dumped = json.dumps(top)
    assert "synthetic fixture answer" not in dumped and "Emotion:" not in dumped and "Active Values" not in dumped
