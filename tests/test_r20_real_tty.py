"""R20 real terminal: the REAL ``RealTerminal`` and the REAL ``real_run prereg`` entry on a REAL
pseudo-terminal that is the controlling terminal of the child.  No fake terminal, no seam, no GPU,
no model (the prereg flow never loads one).

The child is started by a double fork (the middle process exits at once), so it is not a
descendant of the test process: the launcher refuses an agent CLI among its ancestors, and a test
run started from one must still work."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import termios
import time

import pytest

from experiments.r20 import prereg
from experiments.r20.guards import repo_root

if not sys.platform.startswith("linux"):  # pragma: no cover
    pytest.skip("needs Linux /dev/tty and /proc", allow_module_level=True)

TIMEOUT = 40.0
PROMPT_END = r"\r?\n> "
SENTENCE = (PROMPT_END, r"(I AM THE HUMAN OPERATOR[^\r\n]*)\r?\n> ")


class Result:
    def __init__(self, output: str, code: int | None, text: str) -> None:
        self.output = output
        self.code = code
        self.text = text  # stdout + stderr when there was no pty


def _environment(extra_path: Path) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("CLAUDE")}
    env["PYTHONPATH"] = str(extra_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_detached(argv: list[str], tmp: Path, *, mode: str, steps: list[tuple[str, str | bytes]] | None = None,
                 cwd: Path | None = None, source: Path | None = None) -> Result:
    """``mode``: ``pty`` (the slave is the controlling terminal), ``pty_no_ctty`` (a pty on
    0/1/2 but NOT the controlling terminal) or ``none`` (/dev/null stdin, files for output)."""

    status = tmp / "status"
    captured = tmp / "captured"
    pidfile = tmp / "pid"
    master, slave = os.openpty() if mode != "none" else (-1, -1)
    env = _environment(source or Path(repo_root()))
    middle = os.fork()
    if middle == 0:  # pragma: no cover - runs in the child
        try:
            if os.fork() != 0:
                os._exit(0)
            os.setsid()  # the grandchild is the session leader (TIOCSCTTY needs one)
            if mode == "pty":
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            if mode == "none":
                null = os.open(os.devnull, os.O_RDONLY)
                out = os.open(captured, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                os.dup2(null, 0)
                os.dup2(out, 1)
                os.dup2(out, 2)
            else:
                for fd in (0, 1, 2):
                    os.dup2(slave, fd)
                os.close(master)
            if slave > 2:
                os.close(slave)
            pidfile.write_text(str(os.getpid()))  # the session leader: the process group to kill
            signal.signal(signal.SIGINT, lambda *_: None)  # Ctrl-C reaches the whole group: this waiter survives it
            target = os.fork()
            if target == 0:
                os.chdir(cwd or Path(repo_root()))
                os.execve(argv[0], argv, env)
            _, wait_status = os.waitpid(target, 0)
            tmp_status = tmp / "status.tmp"
            tmp_status.write_text(str(os.waitstatus_to_exitcode(wait_status)))
            tmp_status.rename(status)  # atomic: the driver never reads half a number
            os._exit(0)
        except BaseException as error:
            (tmp / "child_error").write_text(repr(error))
        finally:
            os._exit(127)
    os.waitpid(middle, 0)
    if slave >= 0:
        os.close(slave)
    output = ""
    deadline = time.monotonic() + TIMEOUT

    def pump(wait: float = 0.2) -> bool:
        nonlocal output
        if master < 0:
            time.sleep(wait)
            return True
        # poll(), not select(): select() raises on a descriptor >= 1024, and a long pytest
        # process (other tests of the same shard leave descriptors open) can reach that.
        poller = select.poll()
        poller.register(master, select.POLLIN | select.POLLHUP)
        if not poller.poll(wait * 1000):
            return True
        try:
            data = os.read(master, 65536)
        except OSError:  # EIO: every slave descriptor is closed
            return False
        if not data:
            return False
        output += data.decode("utf-8", "replace")
        return True

    try:
        offset = 0
        for pattern, reply in steps or []:
            while True:
                match = re.compile(pattern, re.DOTALL).search(output, offset)
                if match:
                    break
                assert pump() and time.monotonic() < deadline, f"prompt {pattern!r} never came; saw {output!r}"
            offset = match.end()
            if isinstance(reply, str):
                reply = (match.group(1) if match.groups() else reply).encode() + b"\n"
            os.write(master, reply)
        while pump():
            if time.monotonic() > deadline:
                raise AssertionError(f"the child never finished; saw {output!r}")
            if master < 0 and status.exists() and status.read_text().strip():
                break
        while not (status.exists() and status.read_text().strip()):
            assert time.monotonic() < deadline, "no exit status"
            time.sleep(0.05)
    finally:
        try:
            os.killpg(int(pidfile.read_text().strip()), signal.SIGKILL)  # the session leader's group
        except (OSError, ValueError):
            pass
        if master >= 0:
            os.close(master)
    code = int(status.read_text().strip())
    return Result(output, code, captured.read_text(errors="replace") if captured.exists() else "")


def python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


PROMPT_SCRIPT = (
    "import sys\n"
    "from experiments.r20.launch import RealTerminal\n"
    "from experiments.r20.guards import GuardRefusal\n"
    "try:\n"
    "    print('GOT[' + RealTerminal().prompt('Name?') + ']')\n"
    "except GuardRefusal as refusal:\n"
    "    print('REFUSED ' + refusal.code, file=sys.stderr)\n"
    "    sys.exit(3)\n"
)


def test_the_real_terminal_prompt_returns_the_typed_text_on_a_real_pty(tmp_path: Path) -> None:
    typed = run_detached(python(PROMPT_SCRIPT), tmp_path, mode="pty", steps=[(PROMPT_END, b"  a typed line \n")])
    assert "Name?" in typed.output
    assert "GOT[a typed line]" in typed.output and typed.code == 0 and "Traceback" not in typed.output


def test_prompt_eof_and_ctrl_c_are_clean_refusals(tmp_path: Path) -> None:
    for name, key, code in (("eof", b"\x04", "CONFIRMATION_EOF"), ("intr", b"\x03", "CONFIRMATION_INTERRUPTED")):
        work = tmp_path / name
        work.mkdir()
        result = run_detached(python(PROMPT_SCRIPT), work, mode="pty", steps=[(PROMPT_END, key)])
        assert f"REFUSED {code}" in result.output and result.code == 3 and "Traceback" not in result.output, result.output


@pytest.mark.parametrize("mode", ["none", "pty_no_ctty"])
def test_prompt_without_a_controlling_terminal_is_a_clean_refusal(tmp_path: Path, mode: str) -> None:
    result = run_detached(python(PROMPT_SCRIPT), tmp_path, mode=mode)
    seen = result.text + result.output
    assert "REFUSED NO_TTY" in seen and result.code == 3 and "Traceback" not in seen


def make_source(tmp: Path) -> Path:
    """A throw-away git checkout of the code (a clean tree, a head), so the pre-registration
    preflight is satisfied without touching the real checkout."""

    root = repo_root()
    target = tmp / "src"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(root / "experiments", target / "experiments", ignore=ignore)
    shutil.copytree(root / "suzka", target / "suzka", ignore=ignore)
    shutil.copy(root / "config.yaml", target / "config.yaml")
    git = ["git", "-C", str(target), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    for command in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "snapshot"]):
        subprocess.run([*git, *command], check=True, capture_output=True)
    return target


def prereg_argv(tmp: Path, revision: str) -> list[str]:
    hf = tmp / "hf"
    refs = hf / "hub" / "models--google--gemma-4-12B-it" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision + "\n")
    return [sys.executable, "-m", "experiments.r20.real_run", "prereg", "--base", str(tmp / "base"),
            "--model-id", "google/gemma-4-12B-it", "--hf-home", str(hf), "--model-revision", revision,
            "--master-seed", "1", "--trial-id", "t", "--human-launch"]


def test_the_prereg_flow_runs_on_a_real_pty_and_records_the_preregistration(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    revision = "0123456789abcdef0123456789abcdef01234567"
    result = run_detached(prereg_argv(tmp_path, revision), tmp_path, mode="pty", steps=[(SENTENCE[1], "")],
                          cwd=source, source=source)
    assert result.code == 0 and "Traceback" not in result.output, result.output
    base = tmp_path / "base"
    path = base / prereg.PREREG_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    content, digest = prereg.load(base)  # hash recorded and verified
    assert "four_unlisted_known_violations" not in content["known_violations"]
    assert (base / prereg.HASH_NAME).read_text().strip() == digest
    assert digest in result.output and revision[:8] in result.output


def test_the_prereg_flow_refuses_cleanly_without_a_terminal_on_eof_and_on_ctrl_c(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    revision = "0123456789abcdef0123456789abcdef01234567"
    argv = prereg_argv(tmp_path, revision)
    none = run_detached(argv, tmp_path, mode="none", cwd=source, source=source)
    assert "real_run refused: NO_TTY" in none.text and none.code not in (0, None) and "Traceback" not in none.text
    for name, key, code in (("eof", b"\x04", "CONFIRMATION_EOF"), ("intr", b"\x03", "CONFIRMATION_INTERRUPTED")):
        work = tmp_path / name
        work.mkdir()
        argv = prereg_argv(work, revision)
        result = run_detached(argv, work, mode="pty", steps=[(PROMPT_END, key)],
                              cwd=source, source=source)
        assert f"real_run refused: {code}" in result.output and result.code not in (0, None), result.output
        assert "Traceback" not in result.output and not (work / "base").exists()
