"""The long script and the persona labels of the r20 smoke (lab only; observations, never verdicts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.r20 import smoke
from experiments.r20.guards import GuardRefusal

SHORT_KEYS = {"ordinal", "label", "seconds", "response_chars", "option_naming", "observed_labels", "model_calls", "sampling",
              "prompt_assembly", "chat_roles", "bare_chat"}


def parse(base: Path, *extra: str) -> Any:
    return smoke.parser().parse_args(["--base", str(base), *extra])


def matrix_results(base: Path) -> dict[str, Any]:
    (path,) = sorted(base.glob("results-smoke-matrix-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_long_script_places_the_probes_at_the_end_and_far_from_the_first_turn() -> None:
    labels = [label for label, _ in smoke.LONG_TURNS]
    texts = [text for _, text in smoke.LONG_TURNS]
    assert len(labels) == smoke.LONG_MAX_TURNS == 16 and len(set(texts)) == 16 and len(set(labels)) == 16
    assert labels[0] == "setup" and labels[1] == "E"  # E stays at its early position
    assert labels[12:] == ["factual", "rephrase", "population", "recall"]  # probes at turns 13-16
    assert 13 - 1 >= 12  # the first probe (turn 13) is at least 12 turns after turn 1
    assert labels[2] == "identity" and all(label.startswith("filler_") for label in labels[3:12])
    assert all(phrase in dict(smoke.LONG_TURNS)["setup"] for phrase in smoke.OPTION_PHRASES)
    assert dict(smoke.LONG_TURNS)["population"] == dict(smoke.TURNS)["population"]
    assert dict(smoke.LONG_TURNS)["rephrase"] == dict(smoke.TURNS)["rephrase"]
    # a filler never names the capital or the option phrases (it would pollute the probes' substring labels)
    fillers = " ".join(text for label, text in smoke.LONG_TURNS if label.startswith("filler_"))
    assert "東京" not in fillers and not any(phrase in fillers for phrase in smoke.OPTION_PHRASES)


def test_the_short_script_and_its_limit_are_unchanged() -> None:
    assert [label for label, _ in smoke.TURNS] == ["setup", "E", "identity", "factual", "rephrase", "population", "recall"]
    assert smoke.MAX_TURNS == 7 and smoke.script_turns("short") is smoke.TURNS


def test_turns_limits_follow_the_script(r20_base: Path) -> None:
    short = parse(r20_base, "--scripted")
    smoke._check_arguments(short)  # noqa: SLF001
    assert short.turns == 7
    long = parse(r20_base, "--scripted", "--script", "long")
    smoke._check_arguments(long)  # noqa: SLF001
    assert long.turns == 16
    for extra in (("--turns", "8"), ("--turns", "0")):
        with pytest.raises(GuardRefusal) as caught:
            smoke._check_arguments(parse(r20_base, "--scripted", *extra))  # noqa: SLF001
        assert caught.value.code == "TURNS_OUT_OF_RANGE"
    smoke._check_arguments(parse(r20_base, "--scripted", "--script", "long", "--turns", "16"))  # noqa: SLF001
    with pytest.raises(GuardRefusal):
        smoke._check_arguments(parse(r20_base, "--scripted", "--script", "long", "--turns", "17"))  # noqa: SLF001
    with pytest.raises(SystemExit):
        parse(r20_base, "--scripted", "--script", "medium")
    assert list(r20_base.iterdir()) == []


def test_lyrical_marker_hits_count_occurrences() -> None:
    assert {"情緒", "穏やか", "静か", "心が", "余韻"} <= set(smoke.LYRICAL_MARKERS)  # the words #414 names
    assert smoke.lyrical_marker_hits("") == 0 and smoke.lyrical_marker_hits("東京です。") == 0
    assert smoke.lyrical_marker_hits("穏やかで静かな余韻、また余韻") == 4


def run_like(label: str, chars: int, hits: int) -> dict[str, Any]:
    return {"turns": [{"label": label, "seconds": 1.0, "response_chars": chars, "lyrical_marker_hits": hits,
                       "option_naming": None, "observed_labels": None, "prompt_assembly": None}]}


def test_aggregate_carries_the_persona_keys_per_label_and_overall() -> None:
    table = smoke.aggregate([run_like("filler_season", 100, 4), run_like("filler_season", 200, 2), run_like("identity", 30, 0)])
    assert table["per_label"]["filler_season"]["mean_lyrical_marker_hits"] == 3.0
    assert table["per_label"]["filler_season"]["mean_response_chars"] == 150.0
    assert table["persona"] == {"kind": "observed", "turns": 3, "mean_lyrical_marker_hits": 2.0, "mean_response_chars": 110.0}
    assert set(table) == {"kind", "runs", "per_label", "persona"}
    assert smoke.aggregate([])["persona"]["mean_lyrical_marker_hits"] is None


def test_the_table_shows_the_persona_columns() -> None:
    item = {"id": "X9", "description": "all (first-user)", "completed_runs": 5, "subjects_planned": 5,
            "key_counts": {title: None for title, _l, _k in smoke.KEY_COLUMNS}, "min_free_bytes": 3 * 1024**3,
            "mean_input_chars": 1200.0, "persona": {"mean_lyrical_marker_hits": 1.25, "mean_response_chars": 210.4}}
    header, row = smoke.matrix_table([item])
    assert "lyric hits" in header and "mean chars" in header and "min free GiB" in header
    assert "1.25" in row and "210" in row and "3.0" in row


def test_a_long_scripted_matrix_with_temperature_runs_end_to_end(
    r20_base: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    args = parse(r20_base, "--scripted", "--matrix", "--record-model-calls", "--repeats", "1", "--script", "long",
                 "--temperature", "0.7", "--matrix-conditions", "X0,X9,X10")
    code = smoke._smoke(args, attest=lambda *a, **k: pytest.fail("no attestation"), gpu_runner=None)  # noqa: SLF001
    out = capsys.readouterr().out
    top = matrix_results(r20_base)
    assert code == 0 and top["requested"]["script"] == "long" and top["requested"]["turns"] == 16
    assert top["generation"]["temperature_requested"] == 0.7 and top["generation"]["do_sample_override"] is True
    assert top["completed_conditions"] == ["X0", "X9", "X10"]
    assert "temperature 0.7" in out and "script long" in out and "lyric hits" in out and "mean chars" in out
    for item in top["conditions"]:
        run = item["runs"][0]
        assert [t["label"] for t in run["turns"]] == [label for label, _ in smoke.LONG_TURNS]
        assert all("lyrical_marker_hits" in t and t["response_chars"] > 0 for t in run["turns"])
        assert item["persona"]["turns"] == 16 and item["persona"]["mean_response_chars"] > 0
        assert item["persona"]["mean_lyrical_marker_hits"] == 0.0  # the fixture answer holds no marker word
        assert {"E one", "reph tokyo", "pop number", "recall both"} == set(item["key_counts"])
    assert "synthetic fixture answer" not in json.dumps(top)


def test_the_short_scripted_flow_keeps_its_labels_and_gains_only_the_persona_number(
    r20_base: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert smoke.main(["--base", str(r20_base), "--scripted"]) == 0
    capsys.readouterr()
    (path,) = sorted(r20_base.glob("results-smoke-*.json"))
    top = json.loads(path.read_text(encoding="utf-8"))
    turns = top["runs"][0]["turns"]
    assert [t["label"] for t in turns] == [label for label, _ in smoke.TURNS]
    assert set(turns[0]) == SHORT_KEYS | {"lyrical_marker_hits"}
