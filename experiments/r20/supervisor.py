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
from experiments.r20.provider import ApprovalActor, ProviderMode

READY_TIMEOUT_S = 90.0
_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "HOME", "HF_HOME")


def child_environment(subject: GenesisSubject, parent: dict[str, str] | None = None) -> dict[str, str]:
    """An allow-list environment: the parent's ``SUZKA_*`` variables are not
    inherited (as the lab script's ``env -i``)."""

    source = dict(os.environ) if parent is None else parent
    env = {key: source[key] for key in _PASSTHROUGH if key in source}
    env.update(
        {
            "PYTHONPATH": str(guards.repo_root()),
            "SUZKA_CONFIG_PATH": str(subject.config_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


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
            )
        finally:
            log.close()
        return cls(subject, process)

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
            if reply.get("ready"):
                return reply
            time.sleep(0.1)
        raise GuardRefusal("CHILD_NOT_READY")

    def kill_group(self, sig: int = signal.SIGKILL) -> None:
        try:
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass

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
            if self in ChildProcess._live:
                ChildProcess._live.remove(self)

    def __enter__(self) -> ChildProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


@atexit.register
def _kill_leftovers() -> None:  # pragma: no cover - interpreter exit
    for child in list(ChildProcess._live):
        try:
            os.killpg(child.pgid, signal.SIGKILL)
        except OSError:
            pass


__all__ = ["ChildProcess", "child_environment", "pid_alive_as", "process_start_ticks"]
