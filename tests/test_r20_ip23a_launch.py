"""IP-23a: the Human-launch guard (the central safety property), the spawn
refusals and the child-side checks.  No real model, no GPU, no terminal."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import subprocess
import sys

import pytest
from r20_ip23_helpers import FakeTerminal, human_launch, real_model, refusal_code, sentence

from experiments.r20 import launch
from experiments.r20.factory import Tier, create_subject
from experiments.r20.guards import GuardRefusal, repo_root
from experiments.r20.launch import UNREADABLE, HumanLaunch, _attest, harness_ancestry
from experiments.r20.provider import ApprovalActor, ProviderMode
from experiments.r20.supervisor import ChildProcess

DIGEST = "b" * 64


def attest(**changes: object) -> HumanLaunch:
    arguments: dict[str, object] = {
        "human_flag": True, "terminal": FakeTerminal(), "environ": {}, "ancestors": lambda: [],
    }
    arguments.update(changes)
    return _attest("S0", DIGEST, **arguments)  # type: ignore[arg-type]


def test_a_human_at_a_terminal_who_types_the_sentence_is_attested() -> None:
    terminal = FakeTerminal()
    launched = attest(terminal=terminal)
    assert launched.valid_here() and launched.stage == "S0" and launched.model_digest == DIGEST
    assert sentence("S0", DIGEST) in terminal.prompts[0]  # names the stage and the model digest


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
    own = [("python", ["python", "-m", "experiments.r20.smoke", "run"])]
    assert harness_ancestry(dry) and harness_ancestry(serve)
    assert harness_ancestry(own) == []
    assert refusal_code(attest, ancestors=lambda: dry) == "HARNESS_PARENT"


def test_a_pipe_or_a_script_without_a_terminal_cannot_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real terminal: stdin / stdout / stderr must be a terminal and the answer comes from /dev/tty."""

    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    assert launch.RealTerminal().is_tty() is False
    assert refusal_code(
        _attest, "S0", DIGEST, human_flag=True, terminal=launch.RealTerminal(), environ={},
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
    # this very process is pytest, not ``-m experiments.r20.smoke``
    assert refusal_code(launch.check_launch_record, record, launched.nonce, os.getpid()) == "LAUNCH_PARENT_NOT_THE_LAUNCHER"
    assert refusal_code(launch.check_launch_record, tmp_path / "absent.json", launched.nonce, os.getpid()) == "LAUNCH_RECORD_MISSING"


def test_a_scripted_child_refuses_a_real_model_config(r20_base: Path) -> None:
    """A subject prepared with a real model can only be served by a Human-launched real child."""

    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2, model=real_model())
    done = _serve(subject.root, subject.run_id, "--provider", "scripted")
    assert done.returncode == 2 and "MODEL_CONFIG_MODE_MISMATCH" in done.stderr


# ------------------------------------------------ review round 2 (IP-23a)


def test_an_unreadable_proc_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert harness_ancestry([(UNREADABLE, [UNREADABLE])])
    monkeypatch.setattr(os, "getppid", lambda: 2**30)  # a pid with no /proc entry
    assert launch.ancestor_commands() == [(UNREADABLE, [UNREADABLE])]
    assert refusal_code(attest, ancestors=launch.ancestor_commands) == "HARNESS_PARENT"


def test_the_public_attestation_takes_no_seam_and_production_never_imports_the_private_ones() -> None:
    """AST: ``_attest`` / ``_MINT`` are reached only from their own module and from tests."""

    import ast
    import inspect

    assert set(inspect.signature(launch.attest_human_launch).parameters) == {
        "stage", "model_digest", "human_flag", "revision8", "before_confirm"}
    private = {"_attest", "_MINT"}
    owners = {"_attest": "launch.py", "_MINT": "launch.py"}
    offenders: list[str] = []
    for root in (repo_root() / "experiments",):  # suzka/ never imports experiments (an existing architecture test)
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: set[str] = set()
                if isinstance(node, ast.ImportFrom):
                    names = {alias.name for alias in node.names}
                elif isinstance(node, ast.Attribute):
                    names = {node.attr}
                for name in names & private:
                    if path.name != owners[name]:
                        offenders.append(f"{path}:{name}")
    assert offenders == []


def test_the_model_loader_never_trusts_remote_code() -> None:
    """A hostile snapshot would be code execution: ``from_pretrained`` is called without it, and the
    default is False."""

    for path in (repo_root() / "suzka").rglob("*.py"):
        assert "trust_remote_code" not in path.read_text(encoding="utf-8"), path


# ----------------------------------------- the launch record across a restart


def test_a_restart_accepts_the_same_binding_and_refuses_any_other(tmp_path: Path) -> None:
    launched = human_launch()
    record = tmp_path / "launch.json"
    launch.write_launch_record(record, launched)
    launch.write_launch_record(record, launched)  # the planned SIGKILL restart: same subject, same launch
    other = human_launch(stage="S1")
    assert refusal_code(launch.write_launch_record, record, other) == "LAUNCH_RECORD_MISMATCH"
    foreign = human_launch()
    object.__setattr__(foreign, "pid", foreign.pid + 1)
    assert refusal_code(launch.write_launch_record, record, foreign) == "LAUNCH_RECORD_MISMATCH"
    record.write_text("not json")
    assert refusal_code(launch.write_launch_record, record, launched) == "LAUNCH_RECORD_MISMATCH"


class StubProcess:
    """A successful child that has not exited, until the group is killed."""

    _next = 4_000_000

    command: list[str] = []

    def __init__(self) -> None:
        StubProcess._next += 1
        self.pid = StubProcess._next
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.fixture
def stub_popen(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[StubProcess]]:
    from experiments.r20 import supervisor

    made: list[StubProcess] = []

    def popen(command: list[str], **_kwargs: object) -> StubProcess:
        made.append(StubProcess())
        made[-1].command = command
        made[-1].env = _kwargs.get("env")  # type: ignore[attr-defined]
        assert "--provider" in command and "real" in command
        return made[-1]

    def killpg(pgid: int, _sig: int) -> None:  # NEVER a real signal to a made-up pid
        for process in made:
            if process.pid == pgid:
                process.returncode = -9

    monkeypatch.setattr(supervisor, "pid_alive_as", lambda pid, ticks: any(p.pid == pid and p.returncode is None for p in made))
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor.os, "killpg", killpg)
    yield made
    for process in made:  # no made-up pid may reach the atexit killpg of a real run
        process.returncode = process.returncode if process.returncode is not None else -9
    ChildProcess._live[:] = [c for c in ChildProcess._live if c.process not in made]  # noqa: SLF001


def test_spawn_real_twice_on_one_subject_succeeds_both_times(r20_base: Path, stub_popen: list[StubProcess]) -> None:
    subject = create_subject("S0", base=r20_base, tier=Tier.T2, model=real_model())
    launched = human_launch()
    first = ChildProcess.spawn_real(subject, launched, hf_home=None, turn_cap_s=120.0)
    record = (subject.control_dir / "launch.json").read_text()
    assert first.real and ChildProcess._live.count(first) == 1  # noqa: SLF001
    assert refusal_code(ChildProcess.spawn_real, subject, launched, hf_home=None, turn_cap_s=120.0) == "SECOND_REAL_PROCESS"
    first.kill_group()  # the planned SIGKILL
    second = ChildProcess.spawn_real(subject, launched, hf_home=None, turn_cap_s=120.0)
    assert second.pid != first.pid and (subject.control_dir / "launch.json").read_text() == record
    other = human_launch(stage="S1")
    second.kill_group()
    assert refusal_code(ChildProcess.spawn_real, subject, other, hf_home=None, turn_cap_s=120.0) == "LAUNCH_RECORD_MISMATCH"
    assert len(stub_popen) == 2  # no third child was started


def test_the_s0_kill_and_restart_path_goes_through_spawn_real_twice(
    r20_base: Path, stub_popen: list[StubProcess], monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.r20 import s0

    monkeypatch.setattr(ChildProcess, "wait_ready", lambda self, timeout_s=0: {"ready": True})
    monkeypatch.setattr(s0.factory, "verify_first_start", lambda subject, provider=None: object())
    monkeypatch.setattr(s0.factory, "write_identity_record", lambda subject, record: subject.subject_json)
    session = s0.S0Session(r20_base, profile="S0", model=real_model(), launch=human_launch(), dc={"mode": "refuse"})
    first = session.start()
    session.kill()  # kill_phase: SIGKILL of the group ...
    second = session.start()  # ... then the restart of the SAME subject through spawn_real
    assert second.pid != first.pid and len(stub_popen) == 2 and second.real
    assert session.child is second and len(session._lives) == 2  # noqa: SLF001


# ------------------------------------------------ review round 3 (IP-23a)


def test_a_real_child_never_writes_bytecode(r20_base: Path, stub_popen: list[StubProcess]) -> None:
    subject = create_subject("S0", base=r20_base, tier=Tier.T2, model=real_model())
    ChildProcess.spawn_real(subject, human_launch(), hf_home=None, turn_cap_s=120.0)
    command = stub_popen[0].command
    assert command[1] == "-B" and command[2] == "-m"
    assert stub_popen[0].env["PYTHONDONTWRITEBYTECODE"] == "1"  # type: ignore[attr-defined]
    from experiments.r20.supervisor import child_environment

    assert child_environment(subject, {"PATH": "/bin"})["PYTHONDONTWRITEBYTECODE"] == "1"


def test_an_ancestor_hop_that_cannot_be_read_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = launch._command_words  # noqa: SLF001

    def second_hop_unreadable(pid: int) -> tuple[str, list[str]]:
        calls["n"] += 1
        return real(pid) if calls["n"] == 1 else ("", [])

    monkeypatch.setattr(launch, "_command_words", second_hop_unreadable)
    found = launch.ancestor_commands()
    if calls["n"] >= 2:  # (a machine whose parent has no further ancestor above pid 1 never reaches hop 2)
        assert found[-1] == (UNREADABLE, [UNREADABLE]) and harness_ancestry(found)
