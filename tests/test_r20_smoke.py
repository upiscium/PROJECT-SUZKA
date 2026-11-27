"""The real-model smoke (``experiments.r20.smoke``): the guards, and the scripted (fixture-only) flow
end to end.  No real model, no GPU: the terminal, the GPU probe and the HF cache are fakes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest
from r20_ip23_helpers import MODEL_ID, REVISION, human_launch, make_cache

from experiments.r20 import smoke
from experiments.r20.guards import GuardRefusal
from experiments.r20.supervisor import ChildProcess


def args_for(base: Path, *extra: str) -> object:
    return smoke.parser().parse_args(["--base", str(base), *extra])


def real_args(base: Path, hf_home: Path, *extra: str) -> object:
    return args_for(base, "--model-id", MODEL_ID, "--hf-home", str(hf_home), "--model-revision", REVISION, *extra)


def refusal_code(function: object, *args: object, **kwargs: object) -> str:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except GuardRefusal as refusal:
        return refusal.code
    raise AssertionError("no refusal")


def results_of(base: Path) -> dict:
    (path,) = sorted(base.glob("results-smoke-*.json"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- scripted flow


def test_the_scripted_smoke_runs_end_to_end_and_reports_observations(
    r20_base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = smoke.main(["--base", str(r20_base), "--scripted", "--turns", "4"])
    out = capsys.readouterr().out
    assert code == 0  # the scripted E reply names TWO options: an observation, not a failure
    assert "FIXTURE ONLY" in out and "observations, not a verdict" in out and "RAW REAL-MODEL OUTPUT" in out
    top = results_of(r20_base)
    assert top["fixture_only"] is True and top["verdict"] is None and top["completed_runs"] == 1 and top["stopped"] is None
    data = top["runs"][0]
    assert data["ended"] == "completed"
    assert [turn["label"] for turn in data["turns"]] == ["setup", "E", "identity", "factual"]
    naming = data["turns"][1]["option_naming"]
    assert naming == {"kind": "observed", "options_named": 2, "names_exactly_one": False}
    assert data["turns"][0]["option_naming"] is None
    assert all(turn["seconds"] >= 0 and turn["response_chars"] > 0 for turn in data["turns"])
    assert not ChildProcess._live  # noqa: SLF001 - the child was stopped


def test_results_and_output_hold_no_model_text_and_the_private_file_is_0600(
    r20_base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert smoke.main(["--base", str(r20_base), "--scripted", "--turns", "2"]) == 0
    out = capsys.readouterr().out
    (results,) = r20_base.glob("results-smoke-*.json")
    data = json.loads(results.read_text(encoding="utf-8"))["runs"][0]
    private = Path(data["private_file"])
    assert private.is_file() and stat.S_IMODE(private.stat().st_mode) == 0o600
    assert stat.S_IMODE(private.parent.stat().st_mode) == 0o700 and (private.parent / "RETENTION.txt").is_file()
    rows = [json.loads(line) for line in private.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[1]["response"].startswith("案Aを暫定的に選びました")
    for row in rows:  # full raw text only in the private file
        assert row["response"] not in results.read_text(encoding="utf-8")
        assert row["response"] not in out
        assert row["request"] not in results.read_text(encoding="utf-8")
    for turn in data["turns"]:
        assert set(turn) == {"ordinal", "label", "seconds", "response_chars", "option_naming", "observed_labels",
                             "model_calls", "sampling", "prompt_assembly", "chat_roles", "bare_chat", "lyrical_marker_hits"}
        assert turn["model_calls"] is None and turn["sampling"] is None and turn["prompt_assembly"] is None
        assert turn["chat_roles"] is None
        assert turn["bare_chat"] is None


# ----------------------------------------------------------------------- guards


def test_a_production_base_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke.main(["--base", "/var/lib/suzka/smoke", "--scripted"]) == 2
    assert "PRODUCTION_PATH" in capsys.readouterr().err
    assert not Path("/var/lib/suzka/smoke").exists()


def test_a_real_run_without_the_human_launch_flag_is_refused(r20_base: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    hf = make_cache(tmp_path / "hf")
    assert smoke.main(["--base", str(r20_base), "--model-id", MODEL_ID, "--hf-home", str(hf), "--model-revision", REVISION]) == 2
    assert "HUMAN_LAUNCH_FLAG_MISSING" in capsys.readouterr().err
    assert not list(r20_base.iterdir())  # nothing was created


def test_a_real_run_without_a_terminal_is_refused(
    r20_base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hf = make_cache(tmp_path / "hf")
    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    code = smoke.main(["--base", str(r20_base), "--model-id", MODEL_ID, "--hf-home", str(hf),
                       "--model-revision", REVISION, "--human-launch"])
    assert code == 2 and "NO_TTY" in capsys.readouterr().err
    assert not list(r20_base.iterdir())


def test_the_gpu_gate_refuses_before_any_subject_exists(r20_base: Path, tmp_path: Path) -> None:
    hf = make_cache(tmp_path / "hf")
    low = '{"free": 1073741824, "total": 25769803776, "reserved": 0}'  # 1 GiB free
    seen: list[tuple[object, ...]] = []

    def attest(*arguments: object, **keywords: object) -> object:
        seen.append(arguments)
        return human_launch("SMOKE")

    code = refusal_code(smoke._smoke, real_args(r20_base, hf, "--human-launch"),  # noqa: SLF001
                        attest=attest, gpu_runner=lambda: low)
    assert code == "GPU_FREE_BELOW_MINIMUM" and seen and seen[0][0] == "SMOKE"
    assert not list(r20_base.iterdir())


def test_a_revision_that_is_not_the_pinned_one_is_refused(r20_base: Path, tmp_path: Path) -> None:
    hf = make_cache(tmp_path / "hf", revision="f" * 40)
    code = refusal_code(smoke._smoke, real_args(r20_base, hf, "--human-launch"),  # noqa: SLF001
                        attest=lambda *a, **k: human_launch("SMOKE"), gpu_runner=None)
    assert code == "MODEL_REVISION_MISMATCH"


def test_argument_combinations_are_refused(r20_base: Path) -> None:
    run = smoke._smoke  # noqa: SLF001
    seam = {"attest": lambda *a, **k: human_launch("SMOKE"), "gpu_runner": None}
    assert refusal_code(run, args_for(r20_base), **seam) == "MODEL_ARGUMENTS_REQUIRED"
    assert refusal_code(run, args_for(r20_base, "--scripted", "--model-id", MODEL_ID), **seam) == "SCRIPTED_WITH_MODEL_ARGUMENTS"
    assert refusal_code(run, args_for(r20_base, "--scripted", "--turns", "8"), **seam) == "TURNS_OUT_OF_RANGE"
    for bad in ("0", "6", "-1"):
        assert refusal_code(run, args_for(r20_base, "--scripted", "--repeats", bad), **seam) == "REPEATS_OUT_OF_RANGE"
    for bad in ("-0.1", "1.6", "nan"):
        assert refusal_code(run, args_for(r20_base, "--scripted", "--temperature", bad), **seam) == "TEMPERATURE_OUT_OF_RANGE"
    assert not list(r20_base.iterdir())  # every refusal came before anything was created
    assert refusal_code(run, args_for(r20_base, "--scripted", "--turns", "0"), **seam) == "TURNS_OUT_OF_RANGE"


def test_the_launcher_module_names_the_smoke_and_the_binding_is_the_model_digest() -> None:
    from experiments.r20 import launch
    from experiments.r20.model_config import model_digest
    from r20_ip23_helpers import real_model

    assert launch.LAUNCHER_MODULE == "experiments.r20.smoke"
    assert model_digest(real_model()) != model_digest(real_model(load_in_4bit=False))
    assert model_digest(real_model()) != model_digest(real_model(revision="1" * 40))
    assert len(model_digest(real_model())) == 64
