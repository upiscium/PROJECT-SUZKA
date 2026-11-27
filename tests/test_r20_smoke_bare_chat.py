"""``--bare-chat`` of the smoke (observation only): message construction, user-message extraction against the
REAL prompt builder, skipped paths, flag validation, observation labels and the scripted end to end.
No model, no GPU."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import stat
from typing import Any

import pytest

from experiments.r20 import bare_chat_adapter as bca
from experiments.r20 import chat_role_adapter as cra
from experiments.r20 import smoke, supervisor
from experiments.r20.bare_chat_adapter import BARE_SYSTEM_TEXT, BareChatAdapter, extract_user_message
from experiments.r20.guards import GuardRefusal
from experiments.r20.recording_provider import RecordingProvider, read_calls
from suzka.body import EmotionState
from suzka.persona.prompt_builder import PromptBuilder, split_chat_prompt
from suzka.working_memory_contracts import WorkingMemoryView


def real_chat_texts(message: str) -> tuple[str, str]:
    """(system, user) texts exactly as the production chat path builds them for ``message``."""

    view = WorkingMemoryView(selected=(), decisions=(), projected_bytes=0, item_capacity=8, projection_max_bytes=1024, revision=0)
    prompt = PromptBuilder().build(message, EmotionState(), view, conversation_trace_text="Recent conversation:\n- x")
    split = split_chat_prompt(prompt, message)
    assert split is not None
    return split


class FakeTransformers:
    def __init__(self, *, template: bool = True) -> None:
        self.template = template
        self.calls: list[tuple[str, str]] = []
        self.sent: list[Any] = []
        self.outputs = iter(f"out{index}" for index in range(1, 100))

    def _uses_chat_template(self) -> bool:
        return self.template

    def _chat_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"messages": messages}

    def _generate_from(self, inputs: dict[str, Any]) -> str:
        self.sent.append(inputs["messages"])
        return next(self.outputs)

    def generate_chat(self, system_text: str, user_text: str) -> str:
        self.calls.append((system_text, user_text))
        return "plain reply"


class PlainInner:
    def generate_chat(self, system_text: str, user_text: str) -> str:
        return "plain reply"


def roles_of(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(m["role"], m["content"][0]["text"]) for m in messages]


def test_the_system_text_is_the_named_neutral_constant() -> None:
    assert BARE_SYSTEM_TEXT == "あなたは親切なAIアシスタントです。日本語で簡潔に答えてください。"


def test_the_user_message_is_extracted_exactly_from_the_real_builders_output() -> None:
    for message in ("その都市の人口は", "複数行\nの\n質問", "User: a\nAssistant:", "  spaced  "):
        _system, user = real_chat_texts(message)
        assert user.startswith("Private runtime data") and "Emotion:" in user and "Stored episodic evidence:" in user
        assert extract_user_message(user) == message
    assert "Recent conversation:" in real_chat_texts("x")[1]  # the sample really carries SUZKA's runtime data


def test_extraction_refuses_without_exactly_one_marker_or_with_an_empty_message() -> None:
    _system, user = real_chat_texts("hello")
    assert extract_user_message("garbled") is None
    assert extract_user_message(user + "\n\nUser message:\nsecond") is None  # ambiguous: refused, not guessed
    assert extract_user_message("data\n\nUser message:\n") is None
    assert extract_user_message("data\n\nUser message:\n  \n") is None


def test_history_accumulates_in_order_and_the_system_text_is_the_constant() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner)
    for message in ("m1", "m2", "m3"):
        system, user = real_chat_texts(message)
        adapter.generate_chat(system, user)
    first, second, third = inner.sent
    assert roles_of(first) == [("system", BARE_SYSTEM_TEXT), ("user", "m1")]
    assert roles_of(second) == [("system", BARE_SYSTEM_TEXT), ("user", "m1"), ("assistant", "out1"), ("user", "m2")]
    assert roles_of(third)[1:] == [
        ("user", "m1"), ("assistant", "out1"), ("user", "m2"), ("assistant", "out2"), ("user", "m3"),
    ]
    assert all(len(m["content"]) == 1 and m["content"][0]["type"] == "text" for m in third)
    assert inner.calls == []  # nothing took the unchanged path
    joined = json.dumps(third, ensure_ascii=False)
    assert not any(text in joined for text in ("runtime data", "Emotion", "Recent conversation", "Stored episodic"))


def test_the_report_has_numbers_and_the_history_is_per_instance() -> None:
    inner = FakeTransformers()
    adapter = BareChatAdapter(inner)
    system, user = real_chat_texts("m1")
    adapter.generate_chat(system, user)
    report = adapter.take_record_extra()
    assert report is not None and report["applied"] is True and report["adapter_skipped"] is None
    assert report["messages_sent_count"] == 2 and report["history_pairs"] == 0
    assert report["user_chars_before"] == len(user) and report["user_chars_after"] == len("m1")
    assert report["messages_sent"] == inner.sent[0]
    assert adapter.take_record_extra() is None
    adapter.generate_chat(*real_chat_texts("m2"))
    second = adapter.take_record_extra()
    assert second is not None and second["history_pairs"] == 1 and second["messages_sent_count"] == 4
    fresh_inner = FakeTransformers()
    fresh = BareChatAdapter(fresh_inner)  # a new subject: a new process, a new adapter
    fresh.generate_chat(*real_chat_texts("n1"))
    assert [r for r, _t in roles_of(fresh_inner.sent[0])] == ["system", "user"]


@pytest.mark.parametrize(
    ("make", "text", "reason"),
    [
        (PlainInner, "x\n\nUser message:\ny", cra.NO_PRIVATE_PATH),
        (lambda: FakeTransformers(template=False), "x\n\nUser message:\ny", cra.NO_CHAT_TEMPLATE),
        (FakeTransformers, "garbled", bca.NO_USER_MESSAGE),
        (FakeTransformers, "x\n\nUser message:\n", bca.NO_USER_MESSAGE),
    ],
)
def test_a_skipped_call_is_passed_through_unchanged_and_adds_no_history(make: Any, text: str, reason: str) -> None:
    inner = make()
    adapter = BareChatAdapter(inner)
    assert adapter.generate_chat("SYS", text) == "plain reply"
    report = adapter.take_record_extra()
    assert report is not None and report["applied"] is False and report["adapter_skipped"] == reason
    assert report["messages_sent_count"] == 0 and report["history_pairs"] == 0
    assert report["user_chars_before"] == report["user_chars_after"] == len(text) and "messages_sent" not in report
    if isinstance(inner, FakeTransformers):
        assert inner.calls == [("SYS", text)] and inner.sent == []
        inner.template = True
        adapter.generate_chat(*real_chat_texts("m1"))
        assert [r for r, _t in roles_of(inner.sent[0])] == ["system", "user"]  # still no history


def test_a_generate_call_is_passed_through_and_the_wrapper_probes_like_the_inner() -> None:
    class Generating:
        def generate(self, prompt: str) -> str:
            return "g"

    adapter = BareChatAdapter(Generating())
    assert not hasattr(adapter, "generate_chat")
    assert adapter.generate("P") == "g"
    report = adapter.take_record_extra()
    assert report is not None and report["adapter_skipped"] == cra.NOT_A_CHAT_CALL and report["applied"] is False
    assert hasattr(BareChatAdapter(PlainInner()), "generate_chat")


def test_the_recorder_keeps_the_original_form_and_the_report_under_its_own_key(tmp_path: Path) -> None:
    inner = FakeTransformers()
    recorder = RecordingProvider(BareChatAdapter(inner), tmp_path)
    system, user = real_chat_texts("次")
    assert recorder.generate_chat(system, user) == "out1"
    (row,) = read_calls(tmp_path)
    assert row["system_text"] == system and row["user_text"] == user  # the original two-message form
    assert "chat_roles" not in row
    assert row["bare_chat"]["applied"] is True and row["bare_chat"]["messages_sent"] == inner.sent[0]
    assert stat.S_IMODE((tmp_path / "model_calls.jsonl").stat().st_mode) == 0o600
    assert bca.numbers(row) is not None and "messages_sent" not in bca.numbers(row)  # type: ignore[operator]
    assert bca.numbers({}) is None


def parse(base: Path, *extra: str) -> Any:
    return smoke.parser().parse_args(["--base", str(base), "--scripted", *extra])


def test_flag_validation(r20_base: Path) -> None:
    for extra, code in (
        (["--bare-chat"], "BARE_CHAT_NEEDS_RECORD_MODEL_CALLS"),
        (["--bare-chat", "--chat-roles", "--record-model-calls", "--conversation-trace"], "BARE_CHAT_EXCLUDES_CHAT_ROLES"),
        (["--bare-chat", "--chat-roles"], "BARE_CHAT_EXCLUDES_CHAT_ROLES"),
    ):
        with pytest.raises(GuardRefusal) as caught:
            smoke._check_arguments(parse(r20_base, *extra))
        assert caught.value.code == code
        assert smoke.main(["--base", str(r20_base), "--scripted", *extra]) == smoke.EXIT_REFUSED
    smoke._check_arguments(parse(r20_base, "--bare-chat", "--record-model-calls"))  # no conversation trace needed
    assert list(r20_base.iterdir()) == []  # nothing created


def test_the_spawn_functions_default_to_no_flag() -> None:
    for name in ("spawn", "spawn_real"):
        assert inspect.signature(getattr(supervisor.ChildProcess, name)).parameters["bare_chat"].default is False


def test_the_new_observation_labels_are_plain_substring_checks() -> None:
    population = smoke.observe_reply("population", "どの都市か指定してください")
    assert population["asks_which_city"] is True and population["answers_with_number"] is False
    assert smoke.observe_reply("population", "約1400万人です")["answers_with_number"] is True
    assert smoke.observe_reply("population", "約千万人")["answers_with_number"] is True  # 万
    assert smoke.observe_reply("population", "都市名を教えてください")["asks_which_city"] is True
    assert smoke.observe_reply("population", "約2人")["asks_which_city"] is False
    assert smoke.observe_reply("rephrase", "首都のことです")["mentions_capital_answer"] is True
    assert smoke.observe_reply("rephrase", "案Cを優先する")["mentions_capital_answer"] is False
    old = smoke.observe_reply("rephrase", "東京")
    assert {"mentions_tokyo", "mentions_plan_or_comparison", "mentions_capital_answer", "kind"} == set(old)
    assert set(smoke.observe_reply("population", "x")) == {
        "kind", "mentions_tokyo", "mentions_city_word", "answers_with_number", "asks_which_city",
    }


def test_scripted_end_to_end_reports_labels_only(r20_base: Path) -> None:
    assert smoke.main(["--base", str(r20_base), "--scripted", "--turns", "3", "--record-model-calls", "--bare-chat"]) == 0
    (path,) = sorted(r20_base.glob("results-smoke-*.json"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    top = json.loads(path.read_text(encoding="utf-8"))
    assert top["bare_chat"] == {"requested": True, "applied": True, "turns_bare": 0, "turns_passed_through": 3}
    assert top["chat_roles"]["requested"] is False
    run = top["runs"][0]
    assert run["bare_chat"]["requested"] is True and run["ended"] == "completed"
    for turn in run["turns"]:
        bare = turn["bare_chat"]
        assert bare["adapter_skipped"] == cra.NOT_A_CHAT_CALL and bare["applied"] is False
        assert bare["messages_sent_count"] == 0 and bare["history_pairs"] == 0 and "messages_sent" not in bare
        assert bare["user_chars_before"] == bare["user_chars_after"] > 0
        assert turn["chat_roles"] is None
    aggregate = top["aggregate"]["per_label"]["setup"]["bare_chat"]
    assert aggregate["adapter_skipped"] == {cra.NOT_A_CHAT_CALL: 1} and aggregate["applied"] == {"true": 0, "of": 1}
    assert "synthetic fixture answer" not in json.dumps(top)
    calls = Path(run["model_calls_file"])
    assert stat.S_IMODE(calls.stat().st_mode) == 0o600
    rows = read_calls(calls.parent)
    assert rows and all(row["bare_chat"]["adapter_skipped"] == cra.NOT_A_CHAT_CALL and "chat_roles" not in row for row in rows)


def test_flag_off_has_the_same_keys_with_nothing_requested(r20_base: Path) -> None:
    assert smoke.main(["--base", str(r20_base), "--scripted", "--turns", "2", "--record-model-calls"]) == 0
    (path,) = sorted(r20_base.glob("results-smoke-*.json"))
    top = json.loads(path.read_text(encoding="utf-8"))
    assert top["bare_chat"] == {"requested": False, "applied": False, "turns_bare": 0, "turns_passed_through": 0}
    assert all(turn["bare_chat"] is None for turn in top["runs"][0]["turns"])
    assert top["aggregate"]["per_label"]["setup"]["bare_chat"] is None
    assert all("bare_chat" not in row for row in read_calls(Path(top["runs"][0]["model_calls_file"]).parent))
