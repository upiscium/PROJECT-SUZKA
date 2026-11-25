"""R20 S0: the model-call-layer S1 / S2 normalisation (HD-23, option 1) and the review diagnostics.

The key regression: a RAW fenced reply with free-text evidence_steps goes through the real serve
child, the real bridge.review and the production source gate (``suzka.r15.qualified_proposal``)."""

from __future__ import annotations

from concurrent.futures import Future
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_r20_ip23a_drivers import scripted

from experiments.r20 import bridge
from experiments.r20.guards import GuardRefusal
from experiments.r20.normalizing_provider import (
    NAMES_NAME, RAW_NAME, S1_PREFIX, S2_PREFIX, NormalizingProvider, read_names, registered_phase, wrap_loader,
)
from experiments.r20.provider import ScriptedProvider, r15_answer, r15_protocol_script
from suzka.r15.request_templates import s1_request, s2_request

FREE = ["I looked at E.", "Then I thought."]


def _fenced(phase: str, selected: str) -> str:
    data = json.loads(r15_answer(phase, selected))
    data["evidence_steps"] = list(FREE)
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=1) + "\n```"


def _raw_script() -> tuple[str, ...]:
    script = list(r15_protocol_script())
    script[2] = _fenced("S1", "B")
    script[4] = _fenced("S2", "A")
    return tuple(script)


def test_end_to_end_a_raw_fenced_free_evidence_reply_passes_the_production_source_gate(r20_base: Path) -> None:
    session = scripted(r20_base).new_session("S0", stage="G1", script=_raw_script())
    try:
        session.start()
        session.activate()
        reviews = session.run_r15_protocol()
    finally:
        session.stop()
    assert reviews["S1"]["changed"] is True and reviews["S2"]["changed"] is True
    assert session.reply_normalizations == {
        "E": [], "S1": ["fence_stripped", "evidence_steps_replaced"], "S2": ["fence_stripped", "evidence_steps_replaced"],
    }
    private = session.subject.run_dir / "private"
    raw_rows = [json.loads(line) for line in (private / RAW_NAME).read_text().splitlines()]
    assert [row["phase"] for row in raw_rows] == ["E", "S1", "S2"] and raw_rows[1]["raw"] == _raw_script()[2]
    for name in (RAW_NAME, NAMES_NAME):
        assert stat.S_IMODE((private / name).stat().st_mode) == 0o600
    assert "I looked at E" not in (private / NAMES_NAME).read_text()
    for path in session.subject.evidence_dir.rglob("*"):
        if path.is_file():
            assert "I looked at E" not in path.read_text(errors="ignore")


def test_registered_requests_are_detected_from_the_registry_builders_not_copied_text() -> None:
    s1 = s1_request("q", "a")
    s2 = s2_request(r15_answer("S1", "A"), "q", "a", e2_input="e2")
    assert s1.startswith(S1_PREFIX) and s2.startswith(S2_PREFIX)
    assert registered_phase("header\n" + s1) == "S1" and registered_phase("User message:\n" + s2) == "S2"
    assert registered_phase("hello") is None
    # a registered request that only sits in an earlier memory line is not the current message
    assert registered_phase(f"- memory {s1}\n\nUser: next\nAssistant:") is None
    assert registered_phase(f"- {s1}\n\nUser message:\nplain") is None


class _Inner:
    def __init__(self) -> None:
        self.released = False
        self.attached: list[str] = []

    def generate(self, prompt: str) -> str:
        return "```json\n" + json.dumps({"x": 1}) + "\n```"

    def release(self) -> None:
        self.released = True

    def uses_chat_roles(self) -> bool:
        return True

    def generate_chat(self, system_text: str, user_text: str) -> str:
        return "chat:" + user_text[:3]


def test_the_wrapper_delegates_everything_else_and_only_probes_what_the_inner_has(tmp_path: Path) -> None:
    inner = _Inner()
    wrapped = NormalizingProvider(inner, tmp_path)
    wrapped.release()
    assert inner.released and wrapped.uses_chat_roles() is True
    assert wrapped.generate("not a request").startswith("```json")  # unregistered: untouched, nothing recorded
    assert wrapped.generate_chat("s", "abcdef") == "chat:abc" and not list(tmp_path.iterdir())
    assert getattr(NormalizingProvider(ScriptedProvider(), tmp_path), "generate_chat", None) is None
    with pytest.raises(AttributeError):
        wrapped.no_such_method  # noqa: B018
    built = wrap_loader(lambda settings: inner, tmp_path)(object())
    assert isinstance(built, NormalizingProvider) and built.release.__self__ is inner
    assert read_names(tmp_path) == {}


def test_unparseable_reply_is_returned_unchanged_and_records_no_names(tmp_path: Path) -> None:
    class Bad(_Inner):
        def generate(self, prompt: str) -> str:
            return "```json\n{}\n```"

    text = NormalizingProvider(Bad(), tmp_path).generate(s1_request("q", "a"))
    assert text == "```json\n{}\n```" and read_names(tmp_path) == {"S1": []}


# ------------------------------------------------------------- the review diagnostics


def _state(error: BaseException) -> Any:
    future: Future[Any] = Future()
    future.set_exception(error)
    return SimpleNamespace(
        r15_state_port=SimpleNamespace(narrative=SimpleNamespace(_review_committed_turns=lambda refs: None)),
        agent_runtime=SimpleNamespace(submit=lambda *args: future),
    )


def test_a_failed_review_names_the_root_cause_coarsely_and_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge, "committed_chat_turns", lambda state: [object()] * 3)
    try:
        try:
            raise ValueError("invalid candidate record")
        except ValueError as inner:
            raise RuntimeError("handler_failure") from inner
    except RuntimeError as outer:
        error = outer
    with pytest.raises(bridge.BridgeRefused) as refused:
        bridge.review(_state(error), "S1")
    assert str(refused.value) == "REVIEW_FAILED: ValueError: invalid candidate record"
    long = RuntimeError("x" * 500)
    with pytest.raises(bridge.BridgeRefused) as refused:
        bridge.review(_state(long), "S1")
    assert len(str(refused.value).removeprefix("REVIEW_FAILED: ")) == bridge.DETAIL_MAX


def test_the_review_failure_reaches_the_s0_guard_refusal_detail(r20_base: Path) -> None:
    session = scripted(r20_base).new_session("S0", stage="G1")
    session.client = None
    from experiments.r20 import s0

    monkey = pytest.MonkeyPatch()
    monkey.setattr(s0, "send", lambda *a, **k: ({"ok": False, "error": "REVIEW_FAILED: ValueError: invalid candidate record"}, 1))
    try:
        with pytest.raises(GuardRefusal) as refused:
            session.review("S1")
    finally:
        monkey.undo()
    assert refused.value.code == "R15_REVIEW_REFUSED" and "invalid candidate record" in refused.value.detail
