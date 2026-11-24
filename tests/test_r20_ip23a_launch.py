"""IP-23a: the Human-launch guard (the central safety property), the spawn
refusals and the child-side checks.  No real model, no GPU, no terminal."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
from r20_ip23_helpers import FakeTerminal, human_launch, real_model, refusal_code, sentence

from experiments.r20 import launch
from experiments.r20.factory import Tier, create_subject
from experiments.r20.guards import GuardRefusal, repo_root
from experiments.r20.launch import HumanLaunch, attest_human_launch, harness_ancestry
from experiments.r20.provider import ApprovalActor, ProviderMode
from experiments.r20.supervisor import ChildProcess

DIGEST = "b" * 64


def attest(**changes: object) -> HumanLaunch:
    arguments: dict[str, object] = {
        "human_flag": True, "terminal": FakeTerminal(), "environ": {}, "ancestors": lambda: [],
    }
    arguments.update(changes)
    return attest_human_launch("S0", DIGEST, **arguments)  # type: ignore[arg-type]


def test_a_human_at_a_terminal_who_types_the_sentence_is_attested() -> None:
    terminal = FakeTerminal()
    launched = attest(terminal=terminal)
    assert launched.valid_here() and launched.stage == "S0" and launched.prereg_sha256 == DIGEST
    assert sentence("S0", DIGEST) in terminal.prompts[0]  # names the stage and the pre-registration hash


def test_every_refusal_of_the_guard() -> None:
    assert refusal_code(attest, human_flag=False) == "HUMAN_LAUNCH_FLAG_MISSING"
    assert refusal_code(attest, terminal=FakeTerminal(tty=False)) == "NO_TTY"
    assert refusal_code(attest, terminal=FakeTerminal(foreground=False)) == "NOT_FOREGROUND"
    assert refusal_code(attest, environ={"CLAUDECODE": "1"}) == "AGENT_ENVIRONMENT"
    assert refusal_code(attest, environ={"CLAUDE_CODE_ENTRYPOINT": "cli"}) == "AGENT_ENVIRONMENT"
    assert refusal_code(attest, ancestors=lambda: [("claude", ["claude"])]) == "HARNESS_PARENT"
    assert refusal_code(attest, ancestors=lambda: [("node", ["/x/node", "/x/@anthropic-ai/claude-code/cli.js"])]) == "HARNESS_PARENT"
    for typed in ("", "yes", "I AM THE HUMAN OPERATOR", sentence("S1", DIGEST), sentence("S0", "c" * 64), None):
        assert refusal_code(attest, terminal=FakeTerminal(answer=typed or "")) == "CONFIRMATION_MISMATCH"


def test_another_harness_entry_point_as_the_parent_is_refused_but_the_launcher_is_not() -> None:
    dry = [("python", [sys.executable, "-m", "experiments.r20.dry_run"])]
    serve = [("python", ["python", "-m", "experiments.r20.serve", "/x"])]
    own = [("python", ["python", "-m", "experiments.r20.real_run", "run"])]
    assert harness_ancestry(dry) and harness_ancestry(serve)
    assert harness_ancestry(own) == []
    assert refusal_code(attest, ancestors=lambda: dry) == "HARNESS_PARENT"


def test_a_pipe_or_a_script_without_a_terminal_cannot_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real terminal: stdin / stdout / stderr must be a terminal and the answer comes from /dev/tty."""

    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    assert launch.RealTerminal().is_tty() is False
    assert refusal_code(
        attest_human_launch, "S0", DIGEST, human_flag=True, terminal=launch.RealTerminal(), environ={},
        ancestors=lambda: [],
    ) == "NO_TTY"


def test_the_real_terminal_reads_its_answer_from_dev_tty_not_from_stdin() -> None:
    import inspect

    source = inspect.getsource(launch.RealTerminal.prompt)
    assert "/dev/tty" in source and "sys.stdin" not in source and "input(" not in source


def test_there_is_no_switch_that_turns_a_check_off() -> None:
    source = Path(launch.__file__).read_text(encoding="utf-8")
    for name in ("environ.get(", "os.getenv", "R20_SKIP", "SUZKA_R20_ALLOW", "--force", "--yes", "bypass"):
        assert name not in source
    import experiments.r20.real_run as real_run

    flags = {action.option_strings[0] for sub in real_run.parser()._subparsers._group_actions[0].choices.values()  # noqa: SLF001
             for action in sub._actions if action.option_strings}  # noqa: SLF001
    assert not {flag for flag in flags if any(word in flag for word in ("force", "yes", "skip", "no-tty", "confirm-launch"))}


def test_a_launch_cannot_be_forged_or_borrowed() -> None:
    with pytest.raises(GuardRefusal) as forged:
        HumanLaunch("S0", DIGEST, "n", os.getpid(), 0.0, object())
    assert forged.value.code == "HUMAN_LAUNCH_FORGED"
    borrowed = human_launch()
    other = HumanLaunch.__new__(HumanLaunch)
    assert borrowed.valid_here() and not isinstance(other, type(None))
    elsewhere = attest()
    object.__setattr__(elsewhere, "pid", os.getpid() + 1)  # as if minted by another process
    assert not elsewhere.valid_here()


# ------------------------------------------------------- spawn refusals


def test_spawn_refuses_a_real_provider_for_every_non_human_caller(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    for provider in (ProviderMode.REAL,):
        for actor in (ApprovalActor.SCRIPTED, ApprovalActor.HUMAN):
            code = refusal_code(ChildProcess.spawn, subject, provider=provider, actor=actor)
            assert code in {"SCRIPTED_APPROVAL_WITH_REAL_PROVIDER", "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH"}
    assert not ChildProcess._live  # noqa: SLF001


def test_spawn_real_needs_this_processes_human_launch(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2, model=real_model())
    assert refusal_code(ChildProcess.spawn_real, subject, None, hf_home=None, turn_cap_s=120.0) == "HUMAN_LAUNCH_REQUIRED"
    foreign = attest()
    object.__setattr__(foreign, "pid", os.getpid() + 1)
    assert refusal_code(ChildProcess.spawn_real, subject, foreign, hf_home=None, turn_cap_s=120.0) == "HUMAN_LAUNCH_REQUIRED"
    assert not ChildProcess._live  # noqa: SLF001


def test_a_second_real_process_is_refused(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2, model=real_model())

    class Alive:
        real = True

        def alive(self) -> bool:
            return True

    fake = Alive()
    ChildProcess._live.append(fake)  # type: ignore[arg-type]  # noqa: SLF001
    try:
        assert refusal_code(ChildProcess.spawn_real, subject, human_launch(), hf_home=None, turn_cap_s=120.0) == "SECOND_REAL_PROCESS"
    finally:
        ChildProcess._live.remove(fake)  # type: ignore[arg-type]  # noqa: SLF001


# ----------------------------------------------------- the child's own checks


def _serve(root: Path, run_id: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(repo_root())}
    return subprocess.run(
        [sys.executable, "-m", "experiments.r20.serve", str(root), "--run-id", run_id, *arguments],
        capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL, env=env, cwd=root, check=False,
    )


def test_the_child_refuses_a_real_provider_started_by_anything_but_the_launcher(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2, model=real_model())
    done = _serve(subject.root, subject.run_id, "--provider", "real", "--approval-actor", "HUMAN", "--human-launch")
    assert done.returncode == 2 and "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH" in done.stderr  # no tty on stdin
    scripted = _serve(subject.root, subject.run_id, "--provider", "real", "--approval-actor", "SCRIPTED", "--human-launch")
    assert "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER" in scripted.stderr


def test_a_launch_record_must_match_the_nonce_the_parent_and_the_launcher(tmp_path: Path) -> None:
    launched = human_launch()
    record = tmp_path / "launch.json"
    launch.write_launch_record(record, launched)
    assert refusal_code(launch.check_launch_record, record, None, os.getppid()) == "LAUNCH_NONCE_MISSING"
    assert refusal_code(launch.check_launch_record, record, "wrong", os.getpid()) == "LAUNCH_RECORD_MISMATCH"
    assert refusal_code(launch.check_launch_record, record, launched.nonce, os.getppid()) == "LAUNCH_PARENT_MISMATCH"
    # this very process is pytest, not ``-m experiments.r20.real_run``
    assert refusal_code(launch.check_launch_record, record, launched.nonce, os.getpid()) == "LAUNCH_PARENT_NOT_THE_LAUNCHER"
    assert refusal_code(launch.check_launch_record, tmp_path / "absent.json", launched.nonce, os.getpid()) == "LAUNCH_RECORD_MISSING"


def test_a_scripted_child_refuses_a_real_model_config(r20_base: Path) -> None:
    """A subject prepared with a real model can only be served by a Human-launched real child."""

    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2, model=real_model())
    done = _serve(subject.root, subject.run_id, "--provider", "scripted")
    assert done.returncode == 2 and "MODEL_CONFIG_MODE_MISMATCH" in done.stderr
