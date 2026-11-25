"""R20 S0: the COMPROMISED, experiment-side-only S1 / S2 reply normalisation (HD-23, a').
No model, no process: the chat route is replaced with fixed replies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from r20_ip23_helpers import make_cache, make_prereg
from test_r20_ip23a_drivers import _answering, real_context, scripted

from experiments.r15_p1.protocol import parse_candidate
from experiments.r20 import prereg
from experiments.r20.guards import GuardRefusal
from experiments.r20.report import build_report
from experiments.r20.reply_normalizer import normalize_reply

FREE = ["I looked at E.", "Then I thought.", "Then more.", "Finally I answered."]


def _record(phase: str = "S1", steps: Any = None) -> dict[str, Any]:
    return {
        "schema": 1, "phase": phase, "assumed_referent": "A", "assessment": "SETTLED",
        "uncertainty": "LOW", "evidence_steps": steps if steps is not None else ["E"], "rationale": "because",
    }


def _valid_choices() -> tuple[str, str, str]:
    import experiments.r15_p1.protocol as p

    return (sorted(p._CHOICES)[0], sorted(p._ASSESSMENTS)[0], sorted(p._UNCERTAINTIES)[0])


def _good(phase: str, steps: Any) -> dict[str, Any]:
    referent, assessment, uncertainty = _valid_choices()
    return {**_record(phase, steps), "assumed_referent": referent, "assessment": assessment, "uncertainty": uncertainty}


def test_fenced_reply_with_free_text_evidence_steps_is_normalised_and_parses() -> None:
    raw = "```json\n" + json.dumps(_good("S1", FREE)) + "\n```\n"
    text, applied = normalize_reply(raw, "S1")
    assert applied == ["fence_stripped", "evidence_steps_replaced"]
    assert json.loads(text)["evidence_steps"] == ["E"]
    assert parse_candidate(text, phase="S1", has_reveal=False).evidence_steps == ("E",)
    plain = "```\n" + json.dumps(_good("S1", ["E"])) + "```"
    assert normalize_reply(plain, "S1")[1] == ["fence_stripped"]


def test_s2_expected_list_with_reveal_and_model_fields_untouched() -> None:
    record = _good("S2", FREE)
    text, applied = normalize_reply(json.dumps(record), "S2")
    assert applied == ["evidence_steps_replaced"]
    data = json.loads(text)
    assert data["evidence_steps"] == ["E", "S1", "E2"]
    assert {k: v for k, v in data.items() if k != "evidence_steps"} == {k: v for k, v in record.items() if k != "evidence_steps"}
    assert parse_candidate(text, phase="S2", has_reveal=True).evidence_steps == ("E", "S1", "E2")


@pytest.mark.parametrize(
    "raw",
    [
        "not json", "[1, 2]", '"text"', "```json\nnot json\n```", "prefix ```json\n{}\n```",
        json.dumps({**_good("S1", FREE), "extra": 1}),
        json.dumps({k: v for k, v in _good("S1", FREE).items() if k != "rationale"}),
        '{"schema": 1, "schema": 1}', '{"a": NaN}',
    ],
)
def test_anything_not_the_exact_closed_schema_is_returned_unchanged(raw: str) -> None:
    assert normalize_reply(raw, "S1") == (raw, [])


def test_a_canonical_reply_is_byte_identical_and_records_nothing() -> None:
    raw = json.dumps(_good("S1", ["E"]), indent=2)
    assert normalize_reply(raw, "S1") == (raw, [])
    assert normalize_reply("whatever", None) == ("whatever", [])


def _five(s1: str, s2: str) -> list[str]:
    return ["setup reply", "E reply", s1, "E2 reply", s2]


def test_session_records_names_only_and_keeps_raw_in_the_private_turns_file(r20_base: Path) -> None:
    session = real_context(r20_base, gpu_check=lambda: None).new_session("S0", stage="G1")
    s1_raw = "```json\n" + json.dumps(_good("S1", FREE)) + "\n```"
    s2_raw = json.dumps(_good("S2", FREE))
    _answering(session, _five(s1_raw, s2_raw))
    session.review = lambda stage: {"stage": stage}  # type: ignore[method-assign]
    session.run_r15_protocol()
    assert session.reply_normalizations == {
        "S1": ["fence_stripped", "evidence_steps_replaced"], "S2": ["evidence_steps_replaced"],
    }
    rows = [json.loads(line) for line in (session.capture_dir / "p1.turns.jsonl").read_text().splitlines()]
    assert rows[2]["response"] == s1_raw and rows[2]["normalizations"] == ["fence_stripped", "evidence_steps_replaced"]
    assert rows[4]["response"] == s2_raw and rows[0]["normalizations"] == []
    from experiments.r20.drivers import measurements

    recorded = json.dumps(measurements(session)["reply_normalizations"])
    assert "I looked at E" not in recorded and "fence_stripped" in recorded
    for path in session.subject.evidence_dir.rglob("*"):
        if path.is_file():
            assert "Finally I answered" not in path.read_text(errors="ignore")


def test_dry_run_path_normalises_and_unparseable_still_refuses(r20_base: Path) -> None:
    plain = scripted(r20_base).new_session("S0", stage="G1")
    _answering(plain, _five(json.dumps(_good("S1", FREE)), json.dumps(_good("S2", FREE))))
    plain.review = lambda stage: {}  # type: ignore[method-assign]
    plain.run_r15_protocol()
    assert plain.reply_normalizations["S1"] == ["evidence_steps_replaced"]
    assert not list(plain.subject.run_dir.rglob("p1.turns.jsonl"))
    bad = scripted(r20_base).new_session("S0", stage="G1")
    _answering(bad, _five("```json\n{}\n```", "x"))
    bad.review = lambda stage: {}  # type: ignore[method-assign]
    with pytest.raises(GuardRefusal) as refused:
        bad.run_r15_protocol()
    assert refused.value.code == "P1_PROTOCOL_REFUSED" and bad.reply_normalizations["S1"] == []


def test_the_compromise_is_in_not_claimed_and_in_the_report(tmp_path: Path) -> None:
    assert "COMPROMISE (HD-23, a')" in prereg.NOT_CLAIMED
    assert "evidence enumeration is NOT measured" in prereg.NOT_CLAIMED
    base = tmp_path / "base"
    make_prereg(base, make_cache(tmp_path / "hf"))
    assert "COMPROMISE (HD-23, a')" in build_report(base)
