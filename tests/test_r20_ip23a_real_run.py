"""IP-23a: the Human-launch entry point.  Every refusal is proved to happen BEFORE
a subject, a child or a model exists; the happy path runs with a fake terminal, a
fake GPU probe and a fake HF cache and a stub in place of the drivers.  No real
model, no GPU, no production path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any

import pytest
from r20_ip23_helpers import (
    FOUR,
    REVISION,
    FakeTerminal,
    make_cache,
    make_prereg,
    real_model,
    refusal_code,
)

from experiments.r20 import drivers, gpu, prereg, real_run
from experiments.r20.caps import Ledger
from experiments.r20.drivers import HumanApprover
from experiments.r20.guards import GuardRefusal
from experiments.r20.real_run import RealLock, Stopper, parser
from experiments.r20.real_run import release_stage_b_with as release_stage_b
from experiments.r20.real_run import run_stage_with as run_stage
from experiments.r20.supervisor import ChildProcess

GOOD_GPU = lambda: json.dumps({"free": 20 * gpu.GIB, "total": 24 * gpu.GIB, "reserved": 0})  # noqa: E731


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A pre-registered base, a fake HF cache, and a dispatch that must not run."""

    hf = make_cache(tmp_path / "hf")
    base = tmp_path / "base"
    _content, digest = make_prereg(base, hf)
    reached: list[str] = []

    def dispatch(ctx: drivers.RunContext, stage: str, content: dict[str, Any]) -> dict[str, Any]:
        reached.append(stage)
        raise AssertionError("a refusal must happen before the drivers")

    monkeypatch.setattr(real_run, "dispatch", dispatch)
    monkeypatch.setattr(real_run.factory, "git_head", lambda: "a" * 40)
    monkeypatch.setattr(real_run, "GLOBAL_LOCK", tmp_path / "global.lock")
    return {"base": base, "hf": hf, "digest": digest, "reached": reached, "tmp": tmp_path}


def args(base: Path, stage: str = "G1", *extra: str) -> Any:
    return parser().parse_args(["run", "--base", str(base), "--stage", stage, *extra])


def launch(lab: dict[str, Any], stage: str = "G1", *extra: str, **seams: Any) -> int:
    seams.setdefault("terminal", FakeTerminal())
    seams.setdefault("environ", {})
    seams.setdefault("gpu_runner", GOOD_GPU)
    seams.setdefault("ancestors", lambda: [])  # this test process runs under an agent CLI, which the guard refuses
    return run_stage(args(lab["base"], stage, "--human-launch", *extra), **seams)


def assert_nothing_started(lab: dict[str, Any]) -> None:
    assert lab["reached"] == []
    assert not ChildProcess._live  # noqa: SLF001
    assert [p.name for p in lab["base"].iterdir() if p.is_dir()] == []  # no subject directory was created


# ------------------------------------------------------------------- refusals


def test_every_refusal_happens_before_anything_starts(lab: dict[str, Any]) -> None:
    base = lab["base"]
    assert refusal_code(run_stage, args(base, "G1"), terminal=FakeTerminal()) == "HUMAN_LAUNCH_FLAG_MISSING"
    assert refusal_code(launch, lab, terminal=FakeTerminal(tty=False)) == "NO_TTY"
    assert refusal_code(launch, lab, terminal=FakeTerminal(foreground=False)) == "NOT_FOREGROUND"
    assert refusal_code(launch, lab, environ={"CLAUDECODE": "1"}) == "AGENT_ENVIRONMENT"
    assert refusal_code(launch, lab, terminal=FakeTerminal(answer="launch")) == "CONFIRMATION_MISMATCH"
    assert_nothing_started(lab)


def test_the_gpu_and_the_model_are_checked_before_anything_starts(lab: dict[str, Any]) -> None:
    assert refusal_code(launch, lab, gpu_runner=lambda: json.dumps({"error": "CUDA_UNAVAILABLE"})) == "CUDA_UNAVAILABLE"
    assert refusal_code(launch, lab, gpu_runner=lambda: json.dumps({"free": 13 * gpu.GIB, "total": 24 * gpu.GIB})) == "GPU_FREE_BELOW_MINIMUM"
    assert refusal_code(launch, lab, gpu_runner=lambda: "garbage") == "CUDA_PROBE_FAILED"
    (lab["hf"] / "hub" / "models--example-org--example-12b-it" / "refs" / "main").write_text("e" * 40 + "\n")
    (lab["hf"] / "hub" / "models--example-org--example-12b-it" / "snapshots" / ("e" * 40)).mkdir()
    assert refusal_code(launch, lab) == "MODEL_REVISION_MISMATCH"  # the cache holds another digest than the pinned one
    import shutil

    shutil.rmtree(lab["hf"] / "hub")
    assert refusal_code(launch, lab) == "MODEL_NOT_IN_CACHE"
    assert_nothing_started(lab)


def test_the_caps_refuse_before_the_human_is_even_asked(lab: dict[str, Any]) -> None:
    terminal = FakeTerminal()
    ledger = Ledger(lab["base"])
    assert refusal_code(launch, lab, "G1", "--max-subjects", "43", terminal=terminal) == "CAP_ABOVE_LIMIT"
    assert refusal_code(launch, lab, "G1", "--turn-seconds", "121", terminal=terminal) == "CAP_ABOVE_LIMIT"
    for index in range(42):
        ledger.append(kind="subject", launch="L", stage="F1", run_id=str(index), elapsed_s=1.0)
    assert refusal_code(launch, lab, terminal=terminal) == "CAP_MAX_SUBJECTS"
    assert refusal_code(launch, lab, "G1", "--max-subjects", "5", terminal=terminal) == "CAP_MAX_SUBJECTS"  # lower only
    assert terminal.prompts == []  # nobody was asked to confirm a launch that would be refused
    assert_nothing_started(lab)


def test_the_total_hours_cap_counts_every_earlier_launch(lab: dict[str, Any]) -> None:
    ledger = Ledger(lab["base"])
    for index in range(5):
        ledger.append(kind="end", launch=f"L{index}", elapsed_s=2.5 * 3600)
    assert refusal_code(launch, lab) == "CAP_TOTAL_HOURS"
    assert_nothing_started(lab)


def test_a_stop_file_refuses_a_new_launch(lab: dict[str, Any]) -> None:
    (lab["base"] / real_run.STOP_FILE).write_text("stop")
    assert refusal_code(launch, lab) == "STOP_FILE"
    assert_nothing_started(lab)


def test_a_production_base_the_missing_or_altered_preregistration_and_an_unknown_stage(lab: dict[str, Any], tmp_path: Path) -> None:
    for place in ("/etc/r20", "/opt/project-suzka/r20", "/var/lib/suzka/r20"):
        assert refusal_code(run_stage, args(Path(place), "G1", "--human-launch")) in {"PRODUCTION_PATH"}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert refusal_code(run_stage, args(empty, "G1", "--human-launch"), terminal=FakeTerminal()) == "PREREG_MISSING"
    path = lab["base"] / prereg.PREREG_NAME
    os.chmod(path, 0o600)
    path.write_bytes(path.read_bytes() + b" ")
    os.chmod(path, 0o400)
    assert refusal_code(launch, lab) == "PREREG_ALTERED"
    with pytest.raises(SystemExit):
        parser().parse_args(["run", "--base", "x", "--stage", "NOPE"])
    assert_nothing_started(lab)


def test_a_second_concurrent_real_launcher_is_refused(lab: dict[str, Any]) -> None:
    first = RealLock(lab["base"], lab["tmp"] / "global.lock")
    with first:
        (lab["tmp"] / "b2").mkdir()
        second = RealLock(lab["tmp"] / "b2", lab["tmp"] / "global.lock")
        assert refusal_code(second.__enter__) == "ANOTHER_REAL_PROCESS"
        assert refusal_code(launch, lab, lock=RealLock(lab["base"], lab["tmp"] / "global.lock")) == "ANOTHER_REAL_PROCESS"
    with RealLock(lab["base"], lab["tmp"] / "global.lock"):  # released: another launcher can run now
        pass
    assert_nothing_started(lab)


def test_the_command_line_entry_refuses_without_a_terminal(lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    argv = ["run", "--base", str(lab["base"]), "--stage", "G1", "--human-launch"]
    assert real_run.main(argv) == real_run.EXIT_REFUSED
    assert "NO_TTY" in capsys.readouterr().err
    assert real_run.main(argv[:-1]) == real_run.EXIT_REFUSED  # no --human-launch
    assert "HUMAN_LAUNCH_FLAG_MISSING" in capsys.readouterr().err
    assert_nothing_started(lab)


# ------------------------------------------------------------------ stage B


def test_stage_f1_refuses_without_the_humans_release_file(lab: dict[str, Any]) -> None:
    assert refusal_code(run_stage, args(lab["base"], "F1", "--human-launch"), terminal=FakeTerminal(), environ={},
                        gpu_runner=GOOD_GPU) == "STAGE_B_NOT_RELEASED"
    assert_nothing_started(lab)


def test_the_release_needs_the_same_launch_guard_and_is_bound_to_the_preregistration(lab: dict[str, Any]) -> None:
    base = lab["base"]
    plain = parser().parse_args(["release-stage-b", "--base", str(base)])
    assert refusal_code(release_stage_b, plain, terminal=FakeTerminal(), environ={},
                        ancestors=lambda: []) == "HUMAN_LAUNCH_FLAG_MISSING"
    human = parser().parse_args(["release-stage-b", "--base", str(base), "--human-launch"])
    assert refusal_code(release_stage_b, human, terminal=FakeTerminal(tty=False), environ={},
                        ancestors=lambda: []) == "NO_TTY"
    assert not drivers.stage_b_released(base, lab["digest"])
    path = release_stage_b(human, terminal=FakeTerminal(), environ={}, ancestors=lambda: [])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and drivers.stage_b_released(base, lab["digest"])
    assert not drivers.stage_b_released(base, "0" * 64) and not drivers.stage_b_released(base, None)
    os.chmod(path, 0o644)
    assert not drivers.stage_b_released(base, lab["digest"])  # a loosened file is not the Human's
    os.chmod(path, 0o600)
    path.write_text(json.dumps({"prereg_sha256": lab["digest"], "released_by": "agent"}))
    assert not drivers.stage_b_released(base, lab["digest"])


# ----------------------------------------------------------------- happy path


def test_a_human_launch_runs_the_stage_and_records_the_ledger_and_the_results(lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                                                capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}

    def dispatch(ctx: drivers.RunContext, stage: str, content: dict[str, Any]) -> dict[str, Any]:
        seen.update(ctx=ctx, stage=stage)
        return {"g1": {"label": "GATE_PASSED"}}

    monkeypatch.setattr(real_run, "dispatch", dispatch)
    assert launch(lab, "G1", "--ssh-host", "lab-host") == 0
    ctx = seen["ctx"]
    assert isinstance(ctx.approver, HumanApprover) and ctx.approver.ssh_host == "lab-host"  # never SCRIPTED with a real model
    assert ctx.real and ctx.model.revision == REVISION and ctx.launch.valid_here() and ctx.prereg_sha256 == lab["digest"]
    assert ctx.turn_cap_s == 120.0 and ctx.budget.stage == "G1" and callable(ctx.gpu_check)
    out = capsys.readouterr().out
    assert lab["digest"] in out and "free at launch" in out and "real_run stop" in out
    results = list(lab["base"].glob("results-G1-*.json"))
    assert len(results) == 1 and stat.S_IMODE(results[0].stat().st_mode) == 0o600
    data = json.loads(results[0].read_text())
    assert data["prereg_sha256"] == lab["digest"] and data["stop_reason"] == "completed" and data["g1"]["label"] == "GATE_PASSED"
    ends = [row for row in Ledger(lab["base"]).rows() if row["kind"] == "end"]
    assert ends and ends[-1]["reason"] == "completed"
    assert (lab["base"] / artifacts_marker()).exists()


def artifacts_marker() -> str:
    from experiments.r20.artifacts import BASE_MARKER

    return BASE_MARKER


def test_the_fixed_launch_stops_when_the_stop_file_appears_and_records_the_reason(lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def dispatch(ctx: drivers.RunContext, stage: str, content: dict[str, Any]) -> dict[str, Any]:
        (lab["base"] / real_run.STOP_FILE).write_text("stop")
        time.sleep(20)  # interrupted by the watcher's SIGTERM -> KeyboardInterrupt in this (main) thread
        return {}

    monkeypatch.setattr(real_run, "dispatch", dispatch)
    started = time.monotonic()
    assert launch(lab, "G1") == 1
    assert time.monotonic() - started < 15
    result = json.loads(next(lab["base"].glob("results-G1-*.json")).read_text())
    assert result["stop_reason"] == "STOP_FILE"
    assert [row["reason"] for row in Ledger(lab["base"]).rows() if row["kind"] == "end"] == ["STOP_FILE"]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # the handlers were restored


def test_a_refusal_inside_a_stage_is_recorded_as_its_reason(lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def dispatch(ctx: drivers.RunContext, stage: str, content: dict[str, Any]) -> dict[str, Any]:
        raise GuardRefusal("TURN_CAP_EXCEEDED", "121 s")

    monkeypatch.setattr(real_run, "dispatch", dispatch)
    assert launch(lab, "S0") == 1
    result = json.loads(next(lab["base"].glob("results-S0-*.json")).read_text())
    assert result["stop_reason"] == "TURN_CAP_EXCEEDED" and result["refused"] == "TURN_CAP_EXCEEDED"


def test_the_stop_command_writes_the_stop_file(lab: dict[str, Any]) -> None:
    assert real_run.main(["stop", "--base", str(lab["base"])]) == 0
    assert (lab["base"] / real_run.STOP_FILE).exists()


# ---------------------------------------------------- the whole process group


def test_ctrl_c_and_sigterm_stop_the_whole_process_group_of_a_child(tmp_path: Path) -> None:
    """A child with a grandchild, in its own session: ``Stopper`` kills both."""

    process = subprocess.Popen(
        [sys.executable, "-c", "import subprocess,time,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"],
        start_new_session=True,
    )

    class Stub(ChildProcess):
        def __init__(self) -> None:  # no subject: the control socket does not exist, ``quit`` fails closed
            self.process = process
            self.pid = process.pid
            self.ticks = None
            self.port = 0
            self.real = False
            self.subject = type("S", (), {"control_socket": tmp_path / "none.sock"})()  # type: ignore[assignment]

    stub = Stub()
    ChildProcess._live.append(stub)  # noqa: SLF001
    time.sleep(0.5)
    group = process.pid
    stopper = Stopper(tmp_path)
    try:
        with pytest.raises(KeyboardInterrupt):
            stopper._handler(signal.SIGINT, None)  # noqa: SLF001
        assert process.wait(timeout=20) is not None
        time.sleep(0.3)
        with pytest.raises(ProcessLookupError):
            os.killpg(group, 0)  # nobody is left in the group, grandchild included
    finally:
        if stub in ChildProcess._live:  # noqa: SLF001
            ChildProcess._live.remove(stub)  # noqa: SLF001
        if process.poll() is None:
            process.kill()
        stopper.restore()


def test_the_stopper_watch_thread_does_not_outlive_the_stopper(tmp_path: Path) -> None:
    before = {t.name for t in threading.enumerate()}
    stopper = Stopper(tmp_path)
    stopper.install()
    stopper.restore()
    time.sleep(1.3)
    assert {t.name for t in threading.enumerate()} - before == set()


def test_prereg_through_the_command_line_creates_once_and_demands_the_four(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    four = tmp_path / "four.json"
    four.write_text(json.dumps(FOUR))
    base = tmp_path / "pb"
    argv = ["prereg", "--base", str(base), "--model-id", "example-org/example-12b-it", "--hf-home", str(tmp_path / "hf"),
            "--model-revision", REVISION, "--master-seed", "5", "--trial-id", "t1", "--known-violations-file", str(four)]
    assert real_run.main(argv) == 0
    digest = capsys.readouterr().out.strip()
    assert prereg.load(base)[1] == digest
    assert real_run.main(argv) == real_run.EXIT_REFUSED and "PREREG_EXISTS" in capsys.readouterr().err
    four.write_text(json.dumps(["x"]))
    assert real_run.main([*argv[:2], str(tmp_path / "pb2"), *argv[3:]]) == real_run.EXIT_REFUSED
    assert "FOUR_KNOWN_VIOLATIONS_REQUIRED" in capsys.readouterr().err
    assert real_model().model_id in json.loads((base / prereg.PREREG_NAME).read_text())["model"]["model_id"]


def test_the_watch_stops_a_launch_that_reaches_a_cap_during_a_wait(tmp_path: Path) -> None:
    """The hours are enforced DURING a long wait too, not only at the next subject."""

    from experiments.r20.caps import Caps, Session

    clock = [0.0]
    ledger = Ledger(tmp_path)
    budget = Session(ledger, Caps(), "L1", "S0", clock=lambda: clock[0])
    stopper = Stopper(tmp_path, budget)
    assert stopper.poll_once() is None and stopper.check() is None
    clock[0] = 2.5 * 3600 + 1
    assert stopper.poll_once() == "CAP_SESSION_HOURS"
    assert refusal_code(stopper.check) == "CAP_SESSION_HOURS"
    (tmp_path / real_run.STOP_FILE).write_text("stop")
    assert stopper.poll_once() == "STOP_FILE" and refusal_code(stopper.check) == "STOP_FILE"
    other = Stopper(tmp_path / "elsewhere", budget)
    for _ in range(Stopper.TICK_EVERY):
        other.poll_once()
    assert any(row["kind"] == "tick" and row["elapsed_s"] > 9000 for row in ledger.rows())  # recorded: a killed launcher counts it


def test_the_public_entries_take_no_seam_and_the_code_must_be_the_preregistered_head(lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    assert list(inspect.signature(real_run.run_stage).parameters) == ["args"]
    assert list(inspect.signature(real_run.release_stage_b).parameters) == ["args"]
    monkeypatch.setattr(real_run.factory, "git_head", lambda: "b" * 40)
    assert refusal_code(launch, lab) == "GIT_HEAD_MISMATCH"
    assert_nothing_started(lab)


def test_a_second_s0_launch_after_a_completed_block_is_refused(lab: dict[str, Any]) -> None:
    (lab["base"] / "results-S0-x.json").write_text(json.dumps({"stop_reason": "completed"}))
    assert refusal_code(launch, lab, "S0") == "S0_ALREADY_RUN"
    assert_nothing_started(lab)
