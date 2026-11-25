"""R20 S0: bounded regeneration of an unacceptable S1 / S2 reply (HD-23, option a: RETRY).

``NormalizingProvider`` regenerates from the SAME request up to ``REPLY_ATTEMPTS_MAX`` times; the first
acceptable attempt is stored, the last attempt is returned when none is, and nothing is repaired or
fabricated.  Every attempt's literal text goes only to the private 0600 file."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest
from test_r20_ip23a_drivers import scripted

from experiments.r15_p0.pilot import Case
from experiments.r20 import drivers
from experiments.r20.guards import GuardRefusal
from experiments.r20.normalizing_provider import (
    E_ATTEMPTS_MAX, E_REQUEST, NAMES_NAME, RAW_NAME, REPLY_ATTEMPTS_MAX, NormalizingProvider, acceptable, read_attempts, read_names,
    registered_phase,
)
from suzka.persona.prompt_builder import CURRENT_MESSAGE_HEADING
from experiments.r20.prereg import NOT_CLAIMED
from experiments.r20.provider import r15_answer, r15_protocol_script
from suzka.r15.qualified_proposal import QualifiedProposalUnavailable, _parse_candidate  # private; test-side only
from suzka.r15.request_templates import s1_request, s2_request


def _fenced(text: str) -> str:
    return "```json\n" + text + "\n```"


def _trailing_comma(phase: str, selected: str) -> str:
    """The real-run shape: fenced AND a trailing comma inside the evidence_steps array."""

    data = json.loads(r15_answer(phase, selected))
    body = json.dumps(data, ensure_ascii=False)
    steps = json.dumps(data["evidence_steps"])
    assert steps in body
    return _fenced(body.replace(steps, steps[:-1] + ",]", 1))


def _valid(phase: str, selected: str) -> str:
    return _fenced(r15_answer(phase, selected))


class _Scripted:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def generate(self, prompt: str) -> str:
        index = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        return self.replies[index]


class _ScriptedChat(_Scripted):
    def generate_chat(self, system_text: str, user_text: str) -> str:
        return self.generate(user_text)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_attempt_limits_are_three_for_s1_s2_and_six_for_e() -> None:
    assert REPLY_ATTEMPTS_MAX == 3 and E_ATTEMPTS_MAX == 6


def test_second_attempt_is_accepted_and_both_literal_texts_stay_private(tmp_path: Path) -> None:
    bad, good = _trailing_comma("S1", "B"), _valid("S1", "A")
    inner = _Scripted([bad, good, good])
    text = NormalizingProvider(inner, tmp_path).generate(s1_request("q", "a"))
    assert inner.calls == 2
    assert acceptable(text, "S1") and json.loads(text)["evidence_steps"] == ["E"]
    raw = _rows(tmp_path / RAW_NAME)
    assert [(row["attempt"], row["raw"], row["accepted"]) for row in raw] == [(1, bad, False), (2, good, True)]
    assert read_attempts(tmp_path) == {"S1": {"attempts": 2, "accepted": True}}
    assert read_names(tmp_path) == {"S1": ["fence_stripped"]}
    names_text = (tmp_path / NAMES_NAME).read_text()
    assert "raw" not in names_text and "rationale" not in names_text and "```" not in names_text
    for name in (RAW_NAME, NAMES_NAME):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def test_all_three_attempts_failing_returns_the_last_text_unchanged(tmp_path: Path) -> None:
    replies = [_trailing_comma("S1", "B"), "not json", _trailing_comma("S1", "B")]
    inner = _Scripted(replies + [_valid("S1", "A")])
    text = NormalizingProvider(inner, tmp_path).generate(s1_request("q", "a"))
    assert inner.calls == 3 and text == replies[-1]  # the LAST attempt, never fabricated, never a 4th call
    assert read_attempts(tmp_path) == {"S1": {"attempts": 3, "accepted": False}}
    assert [row["attempt"] for row in _rows(tmp_path / RAW_NAME)] == [1, 2, 3]


def test_a_first_attempt_that_is_valid_makes_no_extra_call(tmp_path: Path) -> None:
    inner = _Scripted([_valid("S2", "B")])
    provider = NormalizingProvider(inner, tmp_path)
    text = provider.generate(s2_request(r15_answer("S1", "A"), "q", "a", e2_input="e2"))
    assert inner.calls == 1 and acceptable(text, "S2")
    assert read_attempts(tmp_path) == {"S2": {"attempts": 1, "accepted": True}}


def test_the_chat_path_retries_too(tmp_path: Path) -> None:
    inner = _ScriptedChat([_trailing_comma("S1", "B"), _valid("S1", "A")])
    text = NormalizingProvider(inner, tmp_path).generate_chat("system", s1_request("q", "a"))
    assert inner.calls == 2 and acceptable(text, "S1")


def test_unregistered_requests_are_never_retried_nor_recorded(tmp_path: Path) -> None:
    inner = _Scripted(["not json", _valid("S1", "A")])
    provider = NormalizingProvider(inner, tmp_path)
    assert provider.generate("hello there") == "not json"
    assert inner.calls == 1 and not list(tmp_path.iterdir())


def test_the_prereg_states_the_retry_compromise_in_words() -> None:
    assert "up to 3 attempts in total (reply_attempts_max)" in NOT_CLAIMED and "number of attempts per phase" in NOT_CLAIMED


# ---------------------------------------------- the harness check agrees with the production gate

SAMPLES = {
    "S1": [
        r15_answer("S1", "A"),
        _valid("S1", "A"),
        _trailing_comma("S1", "B"),
        "{}",
        "[]",
        "not json",
        r15_answer("S1", "A").replace('"E"', '"E", "X"', 1),
        r15_answer("S1", "A").replace('"schema"', '"schema": 1, "schema"', 1),
        r15_answer("S1", "A").replace("{", '{"extra": 1,', 1),
        r15_answer("S1", "A").replace('"schema": 1', '"schema": NaN', 1),
    ],
    "S2": [r15_answer("S2", "B"), _valid("S2", "B"), _trailing_comma("S2", "B"), r15_answer("S1", "A")],
}


@pytest.mark.parametrize("phase", ["S1", "S2"])
def test_acceptable_agrees_with_the_production_source_gate_parser(phase: str) -> None:
    for sample in SAMPLES[phase]:
        try:
            _parse_candidate(sample, phase)
            production = True
        except QualifiedProposalUnavailable:
            production = False
        assert acceptable(sample, phase) is production, sample[:80]
    assert any(acceptable(sample, phase) for sample in SAMPLES[phase])
    assert not all(acceptable(sample, phase) for sample in SAMPLES[phase])


# ---------------------------------------------- end to end through the real serve child


def _retry_script() -> tuple[str, ...]:
    base = list(r15_protocol_script())
    # setup, E answer, S1 (trailing comma, fenced) -> regenerated S1 (fenced, free-text steps), E2 answer, S2
    base[2] = _valid("S1", "B").replace('["E"]', '["I looked at E."]')
    return (base[0], base[1], _trailing_comma("S1", "B"), base[2], base[3], base[4])


def test_end_to_end_a_first_unparseable_s1_reply_is_regenerated_and_reviewed(r20_base: Path) -> None:
    session = scripted(r20_base).new_session("S0", stage="G1", script=_retry_script())
    try:
        session.start()
        session.activate()
        reviews = session.run_r15_protocol()  # the review goes through bridge.review and the production gate
    finally:
        session.stop()
    assert reviews["S1"]["changed"] is True and reviews["S2"]["changed"] is True
    assert session.reply_attempts == {"E": {"attempts": 1, "accepted": True}, "S1": {"attempts": 2, "accepted": True}, "S2": {"attempts": 1, "accepted": True}}
    assert session.reply_normalizations["S1"] == ["fence_stripped", "evidence_steps_replaced"]
    private = session.subject.run_dir / "private"
    raw = _rows(private / RAW_NAME)
    assert [(row["phase"], row["attempt"]) for row in raw] == [("E", 1), ("S1", 1), ("S1", 2), ("S2", 1)]
    assert raw[1]["raw"] == _retry_script()[2]
    for path in session.subject.evidence_dir.rglob("*"):
        if path.is_file():
            assert "I looked at E" not in path.read_text(errors="ignore")
    from experiments.r20 import drivers

    assert drivers.measurements(session)["reply_attempts"]["S1"] == {"attempts": 2, "accepted": True}


def test_end_to_end_three_unparseable_s1_replies_still_end_in_the_p1_refusal(r20_base: Path) -> None:
    bad = _trailing_comma("S1", "B")
    script = (r15_protocol_script()[0], r15_protocol_script()[1], bad, bad, bad)
    session = scripted(r20_base).new_session("S0", stage="G1", script=script)
    try:
        session.start()
        session.activate()
        with pytest.raises(GuardRefusal) as refused:
            session.run_r15_protocol()
    finally:
        session.stop()
    assert refused.value.code == "P1_PROTOCOL_REFUSED"
    assert session.reply_attempts["S1"] == {"attempts": 3, "accepted": False}


# ---------------------------------------------- HD-23 E retry: the stored E has exactly one prior referent

E_BOTH = "前の案（案A）を比較します。案Bを暫定的に選びました。"
E_ONLY_B = "元の回答では案Bを暫定的に選びました"


def test_the_e_request_does_not_depend_on_the_run_id() -> None:
    assert Case("r20-other-run", "B", "forced", 0, "normal").prompts()["E"] == E_REQUEST
    assert registered_phase(E_REQUEST) == "E"
    assert registered_phase(f"memory\n\n{CURRENT_MESSAGE_HEADING}\n{E_REQUEST}") == "E"
    assert registered_phase(f"User: {E_REQUEST}\nAssistant: x\n\n{CURRENT_MESSAGE_HEADING}\nhello") is None
    assert registered_phase(E_REQUEST + " extra") is None
    assert registered_phase(f"memory\n\nUser: {E_REQUEST}\nAssistant:") == "E"  # the built-prompt (generate) shape


def test_e_second_attempt_is_accepted_and_raw_is_private_only(tmp_path: Path) -> None:
    inner = _Scripted([E_BOTH, E_ONLY_B, E_BOTH])
    text = NormalizingProvider(inner, tmp_path).generate(E_REQUEST)
    assert inner.calls == 2 and text == E_ONLY_B
    raw = _rows(tmp_path / RAW_NAME)
    assert [(r["phase"], r["attempt"], r["raw"], r["accepted"]) for r in raw] == [
        ("E", 1, E_BOTH, False), ("E", 2, E_ONLY_B, True)]
    assert read_attempts(tmp_path) == {"E": {"attempts": 2, "accepted": True}}
    assert read_names(tmp_path) == {"E": []}
    assert E_BOTH not in (tmp_path / NAMES_NAME).read_text()


def test_e_all_six_bad_returns_the_last_unchanged_with_no_seventh_call(tmp_path: Path) -> None:
    replies = [E_BOTH, "どちらも選びません", E_BOTH + "!", E_BOTH + "?", "なし", E_BOTH + "。"]
    inner = _ScriptedChat(replies + [E_ONLY_B])
    text = NormalizingProvider(inner, tmp_path).generate_chat("system", E_REQUEST)
    assert inner.calls == 6 and text == replies[-1]
    assert read_attempts(tmp_path) == {"E": {"attempts": 6, "accepted": False}}
    assert [row["attempt"] for row in _rows(tmp_path / RAW_NAME)] == [1, 2, 3, 4, 5, 6]


def test_e_accepted_on_the_fifth_attempt_stores_attempts_five(tmp_path: Path) -> None:
    inner = _Scripted([E_BOTH] * 4 + [E_ONLY_B, E_BOTH])
    text = NormalizingProvider(inner, tmp_path).generate(E_REQUEST)
    assert inner.calls == 5 and text == E_ONLY_B
    assert read_attempts(tmp_path) == {"E": {"attempts": 5, "accepted": True}}


def test_s1_s2_stop_at_three_even_though_e_allows_six(tmp_path: Path) -> None:
    inner = _Scripted([_trailing_comma("S1", "B")] * 6)
    NormalizingProvider(inner, tmp_path).generate(s1_request("q", "a"))
    assert inner.calls == 3
    other = tmp_path / "s2"
    other.mkdir()
    inner2 = _Scripted(["not json"] * 6)
    NormalizingProvider(inner2, other).generate(s2_request(r15_answer("S1", "A"), "q", "a", e2_input="e"))
    assert inner2.calls == 3


def test_e_first_good_makes_one_call_and_other_turns_are_never_retried(tmp_path: Path) -> None:
    inner = _Scripted([E_ONLY_B])
    NormalizingProvider(inner, tmp_path).generate(E_REQUEST)
    assert inner.calls == 1
    other_dir = tmp_path / "x"
    other_dir.mkdir()
    other = _Scripted([E_BOTH, E_ONLY_B])
    setup = Case("r20-x", "B", "forced", 0, "normal").prompts()["setup"]
    assert NormalizingProvider(other, other_dir).generate(setup) == E_BOTH and other.calls == 1
    assert not list(other_dir.iterdir())


def test_the_prereg_states_the_e_retry_compromise() -> None:
    assert "E retry" in NOT_CLAIMED and "exactly one option" in NOT_CLAIMED and "NOT claimed away" in NOT_CLAIMED
    assert "up to 6 times in total (e_reply_attempts_max)" in NOT_CLAIMED
    assert "S1/S2 up to 3 (reply_attempts_max)" in NOT_CLAIMED and "attempts per phase are reported" in NOT_CLAIMED


def test_the_prereg_records_both_caps_and_requires_the_e_key() -> None:
    from experiments.r20.prereg import REQUIRED_KEYS

    assert "e_reply_attempts_max" in REQUIRED_KEYS and "reply_attempts_max" in REQUIRED_KEYS


def test_end_to_end_a_first_e_reply_naming_a_and_b_is_regenerated_and_reviewed(r20_base: Path) -> None:
    base = r15_protocol_script()
    script = (base[0], E_BOTH, base[1], base[2], base[3], base[4])
    session = scripted(r20_base).new_session("S0", stage="G1", script=script)
    try:
        session.start()
        session.activate()
        reviews = session.run_r15_protocol()  # bridge.review -> production gate; used to die on "no single referent"
    finally:
        session.stop()
    assert reviews["S1"]["changed"] is True
    assert session.reply_attempts["E"] == {"attempts": 2, "accepted": True}
    raw = _rows(session.subject.run_dir / "private" / RAW_NAME)
    assert [(r["phase"], r["attempt"]) for r in raw if r["phase"] == "E"] == [("E", 1), ("E", 2)]
    assert raw[0]["raw"] == E_BOTH
    assert drivers.measurements(session)["reply_attempts"]["E"] == {"attempts": 2, "accepted": True}
    for path in session.subject.evidence_dir.rglob("*"):
        if path.is_file():
            assert E_BOTH not in path.read_text(errors="ignore")



# ---------------------------------------------- HD-23: the S0 case target is fixed to A (E2 target != original B)


def test_the_s0_case_target_is_a_and_the_e2_text_announces_a() -> None:
    from experiments.r20.provider import S0_TARGET

    assert S0_TARGET == "A"
    case = Case("r20-any", S0_TARGET, "forced", 0, "normal")
    assert case.target == "A"
    assert "参照先は案Aです" in case.prompts()["reveal"]
    assert "NOTE (HD-23, E2 target)" in NOT_CLAIMED and "6 of 6" in NOT_CLAIMED
    assert "does not differ from the original" in NOT_CLAIMED


def test_end_to_end_scripted_original_b_with_target_a_is_reviewed_at_s2(r20_base: Path) -> None:
    session = scripted(r20_base).new_session("S0", stage="G1", script=r15_protocol_script())
    try:
        session.start()
        session.activate()
        reviews = session.run_r15_protocol()  # bridge.review S2 -> production gate: original B, target A
    finally:
        session.stop()
    assert reviews["S2"]["changed"] is True


def test_end_to_end_scripted_original_a_with_target_a_is_refused_with_the_diagnostic(r20_base: Path) -> None:
    base = list(r15_protocol_script())
    base[1] = "元の回答では案Aを暫定的に選びました"
    base[2] = r15_answer("S1", "A")
    session = scripted(r20_base).new_session("S0", stage="G1", script=tuple(base))
    try:
        session.start()
        session.activate()
        with pytest.raises(GuardRefusal) as refused:
            session.run_r15_protocol()
    finally:
        session.stop()
    assert "report does not differ from the original" in str(refused.value)
