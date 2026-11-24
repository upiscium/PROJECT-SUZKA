"""Parent-side control of one T2 child (D1 sec. 4-1, 4-3, sec. 9).

The child is a normal process in its OWN process group (``start_new_session``),
cwd = the subject root, started with an allow-list environment.  The parent
keeps the pgid and calls ``os.killpg`` in ``stop`` / ``atexit``; no systemd
unit is involved.  Coarse kill points are K1..K6 (``hooks.KILL_POINTS``); the
poll that decides WHEN to kill is 22d's.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, ClassVar

from experiments.r20 import guards
from experiments.r20.control import send
from experiments.r20.factory import GenesisSubject, Tier
from experiments.r20.guards import GuardRefusal
from experiments.r20.launch import HumanLaunch, write_launch_record
from experiments.r20.provider import ApprovalActor, ProviderMode, check_modes

READY_TIMEOUT_S = 90.0
_PASSTHROUGH = ("PATH", "LANG", "LC_ALL")
# a real model needs the CUDA runtime's library path and device selection; no secret
_GPU_PASSTHROUGH = ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")


def child_home(subject: GenesisSubject) -> Path:
    """A disposable HOME under the run directory (never the operator's)."""

    return subject.run_dir / "home"


def child_environment(
    subject: GenesisSubject,
    parent: dict[str, str] | None = None,
    *,
    hf_home: Path | None = None,
    gpu: bool = False,
) -> dict[str, str]:
    """An allow-list environment: the parent's ``SUZKA_*`` variables are not
    inherited (as the lab script's ``env -i``).  HOME and the XDG directories
    are disposable and under the run directory; ``HF_HOME`` is set only when a
    real-model run names it explicitly."""

    source = dict(os.environ) if parent is None else parent
    env = {key: source[key] for key in _PASSTHROUGH if key in source}
    env.update(
        {
            "PYTHONPATH": str(guards.repo_root()),
            "SUZKA_CONFIG_PATH": str(subject.config_path),
            "PYTHONDONTWRITEBYTECODE": "1",  # no new .pyc is written by (or planted for) a run
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HOME": str(child_home(subject)),
            "XDG_CONFIG_HOME": str(child_home(subject) / ".config"),
            "XDG_CACHE_HOME": str(child_home(subject) / ".cache"),
            "XDG_DATA_HOME": str(child_home(subject) / ".local" / "share"),
            "XDG_STATE_HOME": str(child_home(subject) / ".local" / "state"),
        }
    )
    if gpu:
        env.update({key: source[key] for key in _GPU_PASSTHROUGH if key in source})
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    return env


def _die_with_parent() -> None:  # pragma: no cover - runs in the forked child
    """Linux: SIGKILL this child when its parent dies (PR_SET_PDEATHSIG)."""

    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def process_start_ticks(pid: int) -> int | None:
    """Field 22 of ``/proc/<pid>/stat`` (start time), or ``None`` when the
    process is gone or ``/proc`` is unavailable."""

    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name may contain spaces; the fields after the last ")" are fixed.
    fields = text.rsplit(")", 1)[1].split()
    return int(fields[19])


def pid_alive_as(pid: int, ticks: int | None) -> bool:
    """The pid exists, is not a zombie, and is still the process we started."""

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return ticks is None
    if state in {"Z", "X"}:
        return False
    return ticks is None or process_start_ticks(pid) == ticks


class ChildProcess:
    """One started ``serve`` child (a spawn record: pid, ticks, port)."""

    _live: ClassVar[list[ChildProcess]] = []

    def __init__(self, subject: GenesisSubject, process: subprocess.Popen[bytes]) -> None:
        self.subject = subject
        self.process = process
        self.pid = process.pid
        self.ticks = process_start_ticks(process.pid)
        self.port = subject.port
        self.real = False
        ChildProcess._live.append(self)

    @classmethod
    def spawn(
        cls,
        subject: GenesisSubject,
        *,
        provider: ProviderMode = ProviderMode.SCRIPTED,
        actor: ApprovalActor = ApprovalActor.SCRIPTED,
        provider_config: dict[str, Any] | None = None,
        log_path: Path | None = None,
    ) -> ChildProcess:
        check_modes(provider, actor, human_launch=False)  # before any child exists
        if subject.tier is not Tier.T2:
            raise GuardRefusal("NOT_A_T2_SUBJECT", subject.run_id)
        guards.check_marker(subject.root, subject.run_id)
        command = [
            sys.executable, "-m", "experiments.r20.serve", str(subject.root),
            "--run-id", subject.run_id, "--provider", provider.value,
            "--approval-actor", actor.value,
        ]
        if provider_config is not None:
            subject.provider_config.write_text(json.dumps(provider_config), encoding="utf-8")
            os.chmod(subject.provider_config, 0o600)
            command += ["--provider-config", str(subject.provider_config)]
        child_home(subject).mkdir(mode=0o700, exist_ok=True)
        log = open(log_path or subject.run_dir / "serve.log", "ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=subject.root,
                env=child_environment(subject),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,
            )
        finally:
            log.close()
        return cls(subject, process)

    @classmethod
    def spawn_real(
        cls,
        subject: GenesisSubject,
        launch: HumanLaunch,
        *,
        hf_home: Path | None,
        turn_cap_s: float,
        log_path: Path | None = None,
    ) -> ChildProcess:
        """The ONLY way a real-model child starts: it needs the attestation of THIS
        process's Human launch (``attest_human_launch``).  ``spawn`` still refuses a
        real provider for every caller."""

        if not isinstance(launch, HumanLaunch) or not launch.valid_here():
            raise GuardRefusal("HUMAN_LAUNCH_REQUIRED")
        check_modes(ProviderMode.REAL, ApprovalActor.HUMAN, human_launch=True)
        if subject.tier is not Tier.T2:
            raise GuardRefusal("NOT_A_T2_SUBJECT", subject.run_id)
        # exactly ONE real model process at a time (the GPU has no room for two)
        if any(other.real and other.alive() for other in cls._live):
            raise GuardRefusal("SECOND_REAL_PROCESS")
        guards.check_marker(subject.root, subject.run_id)
        # once per subject: a restart (the planned SIGKILL) finds the record and must match its binding
        write_launch_record(subject.control_dir / "launch.json", launch)
        command = [
            sys.executable, "-B", "-m", "experiments.r20.serve", str(subject.root),
            "--run-id", subject.run_id, "--provider", ProviderMode.REAL.value,
            "--approval-actor", ApprovalActor.HUMAN.value, "--human-launch",
            "--launch-nonce", launch.nonce, "--turn-cap-s", str(turn_cap_s),
        ]
        child_home(subject).mkdir(mode=0o700, exist_ok=True)
        log = open(log_path or subject.run_dir / "serve.log", "ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=subject.root,
                env=child_environment(subject, hf_home=hf_home, gpu=True),
                stdin=None,  # the Human's terminal, inherited: the child checks it is a tty
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,
            )
        finally:
            log.close()
        child = cls(subject, process)
        child.real = True
        return child

    @property
    def pgid(self) -> int:
        return self.pid  # start_new_session: the child leads its own group

    def alive(self) -> bool:
        return self.process.poll() is None and pid_alive_as(self.pid, self.ticks)

    def status(self) -> dict[str, Any]:
        reply, server_pid = send(self.subject.control_socket, {"cmd": "status"}, timeout_s=5)
        if server_pid != self.pid or reply.get("pid") != self.pid:
            raise GuardRefusal("CONTROL_PEER_MISMATCH")
        return reply

    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise GuardRefusal("CHILD_EXITED", str(self.process.returncode))
            try:
                reply = self.status()
            except (OSError, ValueError):
                reply = {}
            except GuardRefusal as refusal:
                # a socket still answered by another process (e.g. the killed predecessor's
                # last moments): not ready yet; nothing is sent on this path, and the
                # destination-bound client re-checks the peer before every state change
                if refusal.code != "CONTROL_PEER_MISMATCH":
                    raise
                reply = {}
            if reply.get("ready"):
                return reply
            time.sleep(0.1)
        raise GuardRefusal("CHILD_NOT_READY")

    def kill_group(self, sig: int = signal.SIGKILL) -> None:
        # Only while the child is not yet reaped: afterwards the pgid may belong
        # to an unrelated process.
        if self.process.poll() is None:
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        if self.process.poll() is not None and self in ChildProcess._live:
            ChildProcess._live.remove(self)

    def stop(self) -> None:
        """``quit`` over the control channel, then ``killpg`` in any case."""

        try:
            if self.alive():
                send(self.subject.control_socket, {"cmd": "quit"}, timeout_s=5)
                self.process.wait(timeout=20)
        except (OSError, ValueError, subprocess.TimeoutExpired, GuardRefusal):
            pass
        finally:
            self.kill_group(signal.SIGKILL)

    def __enter__(self) -> ChildProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


@atexit.register
def _kill_leftovers() -> None:  # pragma: no cover - interpreter exit
    for child in list(ChildProcess._live):
        if child.process.poll() is not None:
            continue
        try:
            os.killpg(child.pgid, signal.SIGKILL)
        except OSError:
            pass


__all__ = ["ChildProcess", "child_environment", "pid_alive_as", "process_start_ticks"]
