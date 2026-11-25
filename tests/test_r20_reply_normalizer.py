"""R20 S0: the COMPROMISED, experiment-side-only S1 / S2 reply normalisation (HD-23, a').
Pure function tests; the model-call-layer wiring is in test_r20_normalizing_provider.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from r20_ip23_helpers import make_cache, make_prereg

from experiments.r15_p1.protocol import parse_candidate
from experiments.r20 import prereg
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


def test_the_compromise_is_in_not_claimed_and_in_the_report(tmp_path: Path) -> None:
    assert "COMPROMISE (HD-23, a')" in prereg.NOT_CLAIMED
    assert "evidence enumeration is NOT measured" in prereg.NOT_CLAIMED
    assert "model-call layer BEFORE the subject stores" in prereg.NOT_CLAIMED
    assert "subject's memory holds the normalised text" in prereg.NOT_CLAIMED
    assert "private 0600 file" in prereg.NOT_CLAIMED
    base = tmp_path / "base"
    make_prereg(base, make_cache(tmp_path / "hf"))
    assert "COMPROMISE (HD-23, a')" in build_report(base)
