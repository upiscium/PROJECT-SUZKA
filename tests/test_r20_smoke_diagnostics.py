"""Smoke diagnostics (observation only): repeats, temperature, call recording, reply observations.
Scripted end to end; no real model, no GPU."""

from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest
from r20_ip23_helpers import MODEL_ID, REVISION, human_launch, make_cache

from experiments.r20 import smoke


def results_of(base: Path) -> dict:
    (path,) = sorted(base.glob("results-smoke-*.json"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return json.loads(path.read_text(encoding="utf-8"))


def real_args(base: Path, hf_home: Path, *extra: str) -> object:
    return smoke.parser().parse_args(
        ["--base", str(base), "--model-id", MODEL_ID, "--hf-home", str(hf_home), "--model-revision", REVISION, *extra]
    )


def test_the_option_phrases_are_the_ones_the_setup_prompt_registers() -> None:
    setup = dict(smoke.TURNS)["setup"]
    assert all(phrase in setup for phrase in smoke.OPTION_PHRASES)
    assert len(smoke.TURNS) <= 8 and smoke.MAX_TURNS == len(smoke.TURNS)


def test_reply_observations_are_plain_substring_labels() -> None:
    recall = smoke.observe_reply("recall", "案Aは初期費用を抑える、案Bは付加費用を抑える")
    assert recall == {"kind": "observed", "has_initial_cost_phrase": True, "has_maintenance_cost_phrase": False,
                      "has_both_phrases": False}
    assert smoke.observe_reply("rephrase", "日本の首都は東京です")["mentions_tokyo"] is True
    assert smoke.observe_reply("rephrase", "案Aと案Bを比較すると")["mentions_plan_or_comparison"] is True
    assert smoke.observe_reply("population", "東京は約1400万人")["mentions_tokyo"] is True
    assert smoke.observe_reply("population", "この都市は")["mentions_city_word"] is True
    assert smoke.observe_reply("identity", "私はAIです") is None


def test_repeats_and_recording_end_to_end_with_no_model_text_in_the_results(
    r20_base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = smoke.main(["--base", str(r20_base), "--scripted", "--repeats", "2", "--temperature", "0.3",
                       "--record-model-calls", "--turns", "5"])
    out = capsys.readouterr().out
    assert code == 0
    top = results_of(r20_base)
    text = json.dumps(top, ensure_ascii=False)
    assert top["completed_runs"] == 2 and len(top["runs"]) == 2 and top["stopped"] is None and top["verdict"] is None
    assert top["requested"] == {"repeats": 2, "turns": 5, "record_model_calls": True}
    assert top["generation"]["temperature_requested"] == 0.3 and top["generation"]["override_applied"] is False  # fixture
    assert len({run["run_id"] for run in top["runs"]}) == 2  # fresh subjects
    for run in top["runs"]:
        calls = Path(run["model_calls_file"])
        assert stat.S_IMODE(calls.stat().st_mode) == 0o600 and calls.parent == Path(run["private_file"]).parent
        rows = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
        assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))
        assert rows[0]["method"] == "generate" and "比較用の案は3つです" in rows[0]["prompt"] and rows[0]["output"]
        assert rows[0]["sampling"] is None  # the scripted fixture exposes none
        for row in rows:  # raw prompt / output only in the private file
            assert row["output"] not in text and row["output"] not in out
            assert row["prompt"] not in text
        assert [turn["model_calls"] for turn in run["turns"]] == [1] * 5
        first = run["turns"][0]["prompt_assembly"]
        assert first["kind"] == "observed" and first["messages"] == 1 and first["prior_turn_messages"] == 0
        assert first["prompt_chars"] == len(rows[0]["prompt"]) and first["previous_request_in_input"] is None
        later = run["turns"][4]["prompt_assembly"]
        assert later["trace_turns"] == 0 and later["trace_block_present"] is False
    table = top["aggregate"]
    assert table["runs"] == 2 and table["per_label"]["E"]["observed"]["options_named"] == {"mean": 2.0, "of": 2}
    assert table["per_label"]["setup"]["prompt_assembly"]["prompt_chars"]["of"] == 2
    assert table["per_label"]["rephrase"]["observed"]["mentions_tokyo"] == {"true": 0, "of": 2}
    assert "observations, not a verdict" in out


def test_without_the_flag_no_recording_file_or_wrapper_exists(r20_base: Path) -> None:
    assert smoke.main(["--base", str(r20_base), "--scripted", "--turns", "2"]) == 0
    top = results_of(r20_base)
    assert top["runs"][0]["model_calls_file"] is None and top["requested"]["record_model_calls"] is False
    assert not list(r20_base.rglob("model_calls.jsonl"))


def test_the_spawn_command_is_unchanged_unless_recording_is_asked(r20_base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.r20 import supervisor
    from experiments.r20.factory import Tier, create_subject

    commands: list[list[str]] = []

    class Stop(Exception):
        pass

    def fake_popen(command: list[str], **_keywords: object) -> object:
        commands.append(command)
        raise Stop

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    with pytest.raises(Stop):
        supervisor.ChildProcess.spawn(subject)
    with pytest.raises(Stop):
        supervisor.ChildProcess.spawn(subject, record_calls=True)
    assert "--record-calls" not in commands[0]
    assert commands[1][:-1] == commands[0] and commands[1][-1] == "--record-calls"


def test_the_repeat_loop_stops_at_a_gpu_refusal_and_reports_the_completed_runs(
    r20_base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hf = make_cache(tmp_path / "hf")
    ok = '{"free": 25769803776, "total": 25769803776, "reserved": 0}'
    low = '{"free": 1073741824, "total": 25769803776, "reserved": 0}'
    readings = iter([ok, ok, low])  # attestation check, before run 2, before run 3
    started: list[int] = []

    def fake_run(_args: object, _base: Path, _launch: str, ordinal: int, **_keywords: object) -> tuple[dict, int]:
        started.append(ordinal)
        return {"run": ordinal, "run_id": f"r{ordinal}", "ended": "completed", "turns": [], "serving_after_s": 1.0,
                "gpu": {}, "private_file": "p"}, 0

    monkeypatch.setattr(smoke, "_one_run", fake_run)
    code = smoke._smoke(  # noqa: SLF001
        real_args(r20_base, hf, "--human-launch", "--repeats", "4", "--temperature", "0"),
        attest=lambda *a, **k: human_launch("SMOKE"), gpu_runner=lambda: next(readings),
    )
    assert code == 2 and started == [1, 2]
    top = results_of(r20_base)
    assert top["completed_runs"] == 2 and top["stopped"]["before_run"] == 3
    assert top["stopped"]["code"] == "GPU_FREE_BELOW_MINIMUM"
    assert top["generation"] == {"temperature_requested": 0.0, "temperature_zero_means_greedy": True,
                                 "override_applied": True, "do_sample_override": False, "config_value_kept": False}
    assert top["verdict"] is None
    assert "stopped before run 3" in capsys.readouterr().out
