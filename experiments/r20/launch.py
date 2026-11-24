"""The Human-launch guard of a real-model run (IP-23a; the central safety property).

A real model starts ONLY when a Human, at a real terminal, types an explicit
confirmation sentence.  The guard refuses (a closed ``GuardRefusal`` code):

* ``HUMAN_LAUNCH_FLAG_MISSING``: no ``--human-launch``;
* ``NO_TTY``: stdin, stdout, stderr or ``/dev/tty`` is not a terminal;
* ``NOT_FOREGROUND``: the launcher is not the foreground process group of its
  controlling terminal (a background / detached process);
* ``AGENT_ENVIRONMENT``: an agent's environment variable (``CLAUDE*``) is set;
* ``HARNESS_PARENT``: an ancestor process is an agent CLI or another harness
  entry point (``experiments.r20.*`` other than the launcher);
* ``CONFIRMATION_MISMATCH``: the typed line is not the exact sentence, which
  names the stage and the pre-registration hash.

The confirmation is read from ``/dev/tty``, never from stdin, so a pipe cannot
supply it.  There is NO flag, environment variable or setting that turns a
check off.  ``HumanLaunch`` can be constructed only by ``attest_human_launch``
(a module-private mint), and ``ChildProcess.spawn_real`` requires one that
belongs to THIS process.

Honest limits (KNOWN, NOT PREVENTED): (a) an in-process Python caller that imports the
private seam ``_attest`` (or ``launch._MINT``) can mint a ``HumanLaunch`` with a fake terminal;
(b) a same-uid process driving a pty (script / expect / pty.spawn / setsid -f, a double fork that
also drops the ``CLAUDE*`` variables and the agent ancestry) passes every terminal check.
In detail: a process that can allocate a pseudo-terminal (same uid) can type
the sentence, and can also start ``experiments.r20.serve --provider real`` directly:
the child's own check (a tty on stdin, a launch record whose nonce digest and parent
pid match, a parent command line naming the launcher) needs NO typed sentence, and a
parent can be made to carry that command line.  Python's privacy is by convention
(``_MINT`` is importable).  What the layers stop: a script or pipe without a tty, a
background job, a harness-spawned child, ``ChildProcess.spawn``, a ``S0Session`` or
``spawn_real`` without a ``HumanLaunch``, a scripted approver with a real model.  What
they cannot stop: a deliberate pty-driven launch.  The real protection is the
permission system that forbids the agent to start a real run at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from typing import Protocol

from experiments.r20.guards import GuardRefusal

LAUNCHER_MODULE = "experiments.r20.real_run"
AGENT_ENV_PREFIXES = ("CLAUDE",)
AGENT_COMMANDS = ("claude",)
UNREADABLE = "<unreadable>"
_MINT = object()


def confirmation_sentence(stage: str, prereg_sha256: str, revision8: str = "") -> str:
    """Names the stage, the pre-registration hash and the first 8 characters of the model
    revision, so the Human has to read what is on the screen."""

    tail = f", model revision {revision8}" if revision8 else ""
    return f"I AM THE HUMAN OPERATOR AND I LAUNCH STAGE {stage} OF A REAL-MODEL RUN (pre-registration {prereg_sha256[:12]}{tail})"


class Terminal(Protocol):
    def is_tty(self) -> bool: ...
    def is_foreground(self) -> bool: ...
    def prompt(self, text: str) -> str: ...


class RealTerminal:
    """The controlling terminal.  The prompt goes to, and the answer comes
    from, ``/dev/tty``."""

    def is_tty(self) -> bool:
        try:
            if not all(os.isatty(fd) for fd in (0, 1, 2)):
                return False
            os.close(os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY))
        except OSError:
            return False
        return True

    def is_foreground(self) -> bool:
        try:
            descriptor = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        except OSError:
            return False
        try:
            return os.tcgetpgrp(descriptor) == os.getpgrp()
        except OSError:
            return False
        finally:
            os.close(descriptor)

    def prompt(self, text: str) -> str:
        with open("/dev/tty", "r+", encoding="utf-8") as terminal:
            terminal.write(text + "\n> ")
            terminal.flush()
            return terminal.readline().rstrip("\n")


def agent_environment(environ: Mapping[str, str]) -> list[str]:
    return sorted(name for name in environ if name.startswith(AGENT_ENV_PREFIXES))


def _command_words(pid: int) -> tuple[str, list[str]]:
    try:
        comm = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "", []
    return comm, [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def ancestor_commands() -> list[tuple[str, list[str]]]:
    """``(comm, argv)`` of every ancestor of this process (``/proc`` walk)."""

    found: list[tuple[str, list[str]]] = []
    if not Path(f"/proc/{os.getppid()}/stat").exists():
        return [(UNREADABLE, [UNREADABLE])]  # fail CLOSED: an unreadable /proc is not "no ancestors"
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        words = _command_words(pid)
        if words == ("", []):  # a hop that cannot be read is not "no agent": fail CLOSED
            return [*found, (UNREADABLE, [UNREADABLE])]
        found.append(words)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return [*found, (UNREADABLE, [UNREADABLE])]
    return found


def harness_ancestry(ancestors: list[tuple[str, list[str]]]) -> list[str]:
    """Why an ancestor is an agent CLI or a harness entry point (empty: none)."""

    reasons: list[str] = []
    for comm, argv in ancestors:
        if comm == UNREADABLE:
            reasons.append("ancestry unreadable (fail closed)")
        program = os.path.basename(argv[0]) if argv else ""
        if comm in AGENT_COMMANDS or program in AGENT_COMMANDS or any("claude-code" in word for word in argv[:3]):
            reasons.append(f"agent CLI ({comm or program})")
        if "-m" in argv and any(
            argv[index + 1].startswith("experiments.r20.") and argv[index + 1] != LAUNCHER_MODULE
            for index in range(len(argv) - 1)
            if argv[index] == "-m"
        ):
            reasons.append("harness entry point")
    return reasons


@dataclass(frozen=True, slots=True)
class HumanLaunch:
    """The attestation.  Construct only through ``attest_human_launch``."""

    stage: str
    prereg_sha256: str
    nonce: str
    pid: int
    at: float
    _mint: object

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise GuardRefusal("HUMAN_LAUNCH_FORGED")

    def valid_here(self) -> bool:
        return self.pid == os.getpid()

    def nonce_sha256(self) -> str:
        return hashlib.sha256(self.nonce.encode("utf-8")).hexdigest()


def attest_human_launch(
    stage: str,
    prereg_sha256: str,
    *,
    human_flag: bool,
    revision8: str = "",
    before_confirm: Callable[[Terminal], list[str]] | None = None,
) -> HumanLaunch:
    """The PUBLIC attestation: hard-wired to the real terminal, the real environment and the
    real ancestry; it takes no seam.  ``before_confirm`` may prompt the Human for more input
    and returns the summary lines shown above the confirmation sentence."""

    return _attest(
        stage, prereg_sha256, human_flag=human_flag, terminal=RealTerminal(), environ=dict(os.environ),
        ancestors=ancestor_commands, revision8=revision8, before_confirm=before_confirm,
    )


def _attest(
    stage: str,
    prereg_sha256: str,
    *,
    human_flag: bool,
    terminal: Terminal,
    environ: Mapping[str, str],
    ancestors: Callable[[], list[tuple[str, list[str]]]],
    revision8: str = "",
    before_confirm: Callable[[Terminal], list[str]] | None = None,
) -> HumanLaunch:
    """The seam version.  PRIVATE: only tests import it; production code never does (an AST
    test pins that).  A caller that imports it can mint a launch with a fake terminal: that is a
    documented limit, not a prevented one."""

    if not human_flag:
        raise GuardRefusal("HUMAN_LAUNCH_FLAG_MISSING", "--human-launch is required")
    if not terminal.is_tty():
        raise GuardRefusal("NO_TTY", "a real-model run needs a terminal on stdin, stdout and stderr")
    if not terminal.is_foreground():
        raise GuardRefusal("NOT_FOREGROUND", "the launcher must be the foreground process of its terminal")
    markers = agent_environment(environ)
    if markers:
        raise GuardRefusal("AGENT_ENVIRONMENT", ",".join(markers))
    reasons = harness_ancestry(ancestors())
    if reasons:
        raise GuardRefusal("HARNESS_PARENT", "; ".join(sorted(set(reasons))))
    expected = confirmation_sentence(stage, prereg_sha256, revision8)
    summary = "\n".join(before_confirm(terminal)) + "\n" if before_confirm is not None else ""
    typed = terminal.prompt(f"{summary}Type exactly this line to launch (anything else aborts):\n  {expected}")
    if typed.strip() != expected:
        raise GuardRefusal("CONFIRMATION_MISMATCH")
    return HumanLaunch(stage, prereg_sha256, secrets.token_hex(16), os.getpid(), time.time(), _MINT)


def _binding(launch: HumanLaunch) -> dict[str, object]:
    # What the record binds: this Human launch (its nonce digest, stage, pre-registration) and the
    # launcher process.  A restart of the SAME subject by the SAME launcher presents the same binding.
    return {"nonce_sha256": launch.nonce_sha256(), "launcher_pid": launch.pid, "stage": launch.stage,
            "prereg_sha256": launch.prereg_sha256}


def write_launch_record(path: Path, launch: HumanLaunch) -> None:
    """``control/launch.json`` (0600): what the real ``serve`` child checks.  Written ONCE per
    subject (exclusive create); a restart of the same subject finds it and accepts it only when
    the binding is identical, else it is refused.  The nonce itself is passed on the command
    line; only its digest is stored."""

    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise GuardRefusal("LAUNCH_RECORD_MISMATCH", "unreadable existing record") from error
        if existing != _binding(launch):
            raise GuardRefusal("LAUNCH_RECORD_MISMATCH", "an existing record binds another launch")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(_binding(launch), handle)


def check_launch_record(path: Path, nonce: str | None, parent_pid: int) -> None:
    """The child side: a launch record that matches the nonce, written by the
    launcher that is this child's parent and whose command line is the launcher."""

    if not nonce:
        raise GuardRefusal("LAUNCH_NONCE_MISSING")
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GuardRefusal("LAUNCH_RECORD_MISSING") from error
    if not isinstance(record, dict) or record.get("nonce_sha256") != hashlib.sha256(nonce.encode()).hexdigest():
        raise GuardRefusal("LAUNCH_RECORD_MISMATCH")
    if record.get("launcher_pid") != parent_pid:
        raise GuardRefusal("LAUNCH_PARENT_MISMATCH")
    _comm, argv = _command_words(parent_pid)
    if not (len(argv) >= 3 and "-m" in argv and LAUNCHER_MODULE in argv):
        raise GuardRefusal("LAUNCH_PARENT_NOT_THE_LAUNCHER")


__all__ = [
    "HumanLaunch",
    "RealTerminal",
    "Terminal",
    "agent_environment",
    "ancestor_commands",
    "attest_human_launch",
    "check_launch_record",
    "confirmation_sentence",
    "harness_ancestry",
    "write_launch_record",
]
