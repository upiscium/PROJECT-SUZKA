"""The Human-launch entry point of a real-model run (IP-23a):
``python -m experiments.r20.real_run <command>``.

The Human's commands (on the lab host, in a real terminal; nothing here may be run by an agent):

    python -m experiments.r20.real_run prereg --base B --model-id ORG/NAME --hf-home H \
        --model-revision <40-hex> --master-seed N --trial-id T --human-launch   # you type the four known violations
    python -m experiments.r20.real_run run --base B --stage G1 --human-launch --ssh-host <ssh name of this host>
    python -m experiments.r20.real_run run --base B --stage S0 --human-launch --ssh-host <...>
    python -m experiments.r20.real_run run --base B --stage S1 --human-launch --ssh-host <...>
    python -m experiments.r20.real_run release-stage-b --base B --human-launch      # stage B only
    python -m experiments.r20.real_run stop --base B                                # or Ctrl-C
    python -m experiments.r20.real_run report --base B
    python -m experiments.r20.real_run erase --base B --older-than-days 14 --confirm-erase

Commands:

* ``prereg``         create the immutable pre-registration: the same launch guard, the four known
                     violations typed at the terminal, the whole content shown before you confirm;
* ``run``            one stage (G1 / S0 / S1 / F1) with the real model: needs ``--human-launch``, a
                     terminal, the typed confirmation sentence, a free GPU, a pinned and cached model;
* ``release-stage-b`` the Human's release of stage B (F1), same launch guard;
* ``stop``           write the stop file (the running launcher stops its whole process group);
* ``report``         the Human-readable report from the manifests (no raw text);
* ``erase``          remove the RAW parts of runs (explicit ``--confirm-erase``).

NOTHING in this module starts a model unless every refusal of ``attest_human_launch``,
the caps, the GPU check, the model pin and the one-real-process lock passed.  A process that can drive a pseudo-terminal can still start a child (see the honest limits
in ``experiments/r20/launch.py``); the real protection is that an agent is not permitted to
start a real run at all.  The stage-B release file and the pre-registration are forgeable by
the same uid: they detect, they do not prevent.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid
from typing import Any

from experiments.r20 import artifacts, drivers, factory, gpu, prereg
from experiments.r20.caps import HARD_MAX_SUBJECTS, HARD_TURN_SECONDS, Caps, Ledger, Session
from experiments.r20.drivers import HumanApprover, RunContext, ssh_line
from experiments.r20.guards import GuardRefusal, check_not_production
from experiments.r20.launch import HumanLaunch, attest_human_launch
from experiments.r20.model_config import ModelOverride, check_pinned_revision
from experiments.r20.report import write_report
from experiments.r20.supervisor import ChildProcess

STAGES = ("G1", "S0", "S1", "F1")
STOP_FILE = "STOP"
EXIT_REFUSED = 2
Attest = Callable[..., HumanLaunch]


def user_state_dir() -> Path:
    """Per-user state OUTSIDE any ``--base``: the cumulative ledger and the global lock live here, so
    a new base or a deleted base does not reset the totals (a same-uid process can still edit them)."""

    directory = Path.home() / ".local" / "state" / "suzka-r20"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def default_ledger_path() -> Path:
    return user_state_dir() / "ledger.jsonl"


def default_global_lock() -> Path:
    return user_state_dir() / "real.lock"


def _open_lock(path: Path) -> Any:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)  # never follows a link
    return os.fdopen(descriptor, "a+")


class RealLock:
    """One real-model launcher at a time (a lock under the base AND a per-user one): the GPU has
    no room for two.  ``spawn_real`` also refuses a second live real child in its own process."""

    def __init__(self, base: Path, global_lock: Path | None = None) -> None:
        self._paths = (Path(base) / "real.lock", Path(global_lock) if global_lock else default_global_lock())
        self._handles: list[Any] = []

    def __enter__(self) -> RealLock:
        for path in self._paths:
            handle = _open_lock(path)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                handle.close()
                self.__exit__()
                raise GuardRefusal("ANOTHER_REAL_PROCESS", str(path)) from error
            self._handles.append(handle)
        return self

    def __exit__(self, *exc: object) -> None:
        while self._handles:
            handle = self._handles.pop()
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()


def model_from_prereg(content: dict[str, Any]) -> ModelOverride:
    record = content["model"]
    if record.get("mode") != "real":
        raise GuardRefusal("PREREG_MODEL_NOT_REAL")
    return ModelOverride(
        provider="transformers", model_id=record["model_id"], load_in_4bit=bool(record["load_in_4bit"]),
        max_new_tokens=record.get("max_new_tokens"), device=record["device"], dtype=record["dtype"],
        hf_home=Path(record["hf_home"]) if record.get("hf_home") else None, revision=record["revision_pinned"],
    )


def check_code(content_git_head: str) -> None:
    """The code that runs must be the pre-registered code: a known head, equal to the record, a clean
    working tree (``git status --porcelain``), and the digests of the config template, the request
    templates and the fixtures equal to the recorded ones."""

    head = factory.git_head()
    if head == "unknown":
        raise GuardRefusal("GIT_HEAD_UNKNOWN", "the git head cannot be read")
    if head != content_git_head:
        raise GuardRefusal("GIT_HEAD_MISMATCH", "the code is not the pre-registered git head")
    if factory.git_dirty() is not False:
        raise GuardRefusal("GIT_TREE_DIRTY", "the working tree is not clean (or cannot be read)")


def check_digests(content: dict[str, Any]) -> None:
    current = prereg.code_digests()
    recorded = {
        "config_template_sha256": content["config_template_sha256"],
        "request_templates_sha256": content["request_templates_sha256"],
        "fixtures_sha256": prereg.fixtures_digest(content["fixtures"]),
    }
    differing = sorted(name for name, value in recorded.items() if current[name] != value)
    if differing:
        raise GuardRefusal("CODE_DIGEST_MISMATCH", ",".join(differing))


def preflight(base: Path, caps: Caps, stage: str, prereg_sha256: str, content: dict[str, Any],
              ledger_path: Path) -> None:
    """Refusals BEFORE the Human is asked to confirm anything."""

    ledger = Ledger(ledger_path)
    if ledger.subjects_used() >= caps.max_subjects:
        raise GuardRefusal("CAP_MAX_SUBJECTS", str(caps.max_subjects))
    if ledger.seconds_used() >= caps.total_seconds:
        raise GuardRefusal("CAP_TOTAL_HOURS", f"{ledger.seconds_used() / 3600:.2f} h")
    if (Path(base) / STOP_FILE).exists():
        raise GuardRefusal("STOP_FILE", "remove the stop file deliberately before a new launch")
    if stage == "S0" and any(
        json.loads(path.read_text(encoding="utf-8")).get("stop_reason") == "completed"
        for path in Path(base).glob("results-S0-*.json")
    ):
        raise GuardRefusal("S0_ALREADY_RUN", "K = 3 allows no extra attempt in a second launch")
    check_code(str(content["git_head"]))
    check_digests(content)
    if stage == "F1" and not drivers.stage_b_released(base, prereg_sha256):
        raise GuardRefusal("STAGE_B_NOT_RELEASED", "stage B needs the release file the Human writes (release-stage-b)")


class Stopper:
    """Ctrl-C, SIGTERM and the stop file: stop the WHOLE process group of every live
    child (SIGKILL after a ``quit``), then unwind.  A stopped attempt counts."""

    TICK_EVERY = 60

    def __init__(self, base: Path, budget: Session | None = None) -> None:
        self.base = Path(base)
        self.budget = budget
        self.event = threading.Event()
        self.reason: str | None = None
        self._previous: dict[int, Any] = {}
        self._polls = 0

    def stop_children(self) -> None:
        for child in list(ChildProcess._live):  # noqa: SLF001
            child.stop()

    def _handler(self, signum: int, _frame: object) -> None:
        self.event.set()
        self.stop_children()
        raise KeyboardInterrupt(f"signal {signum}")

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._handler)
        threading.Thread(target=self._watch, name="r20-stopfile", daemon=True).start()

    def poll_once(self) -> str | None:
        """The stop reason now, if any: the stop file, or a cap of the Human reached
        DURING a wait (the hours are also recorded every minute, so a killed launcher
        still counts what it used)."""

        if (self.base / STOP_FILE).exists():
            return "STOP_FILE"
        if self.budget is None:
            return None
        self._polls += 1
        if self._polls % self.TICK_EVERY == 0:
            self.budget.ledger.append(kind="tick", launch=self.budget.launch_id, elapsed_s=round(self.budget.elapsed, 1))
        return self.budget.over()

    def _watch(self) -> None:
        while not self.event.wait(1.0):
            reason = self.poll_once()
            if reason is not None:
                self.reason = reason
                os.kill(os.getpid(), signal.SIGTERM)  # the handler runs in the main thread
                return

    def restore(self) -> None:
        self.event.set()
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)

    def check(self) -> None:
        if self.event.is_set() or (self.base / STOP_FILE).exists():
            raise GuardRefusal("STOP_FILE", "stop requested")
        if self.budget is not None:
            self.budget.check()


def write_results(base: Path, launch_id: str, stage: str, payload: dict[str, Any], prereg_sha256: str) -> Path:
    path = Path(base) / f"results-{stage}-{launch_id}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"stage": stage, "prereg_sha256": prereg_sha256, "launch": launch_id, **payload}, handle,
                  sort_keys=True, indent=1, default=str)
    return path


def _run_stage(args: argparse.Namespace, *, attest: Attest, gpu_runner: Any, lock: RealLock | None,
               ledger_path: Path) -> int:
    """The Human-launched stage.  PRIVATE seam version (``attest`` / ``gpu_runner`` / ``lock`` /
    ``ledger_path``): only tests call it; ``run_stage`` below is hard-wired."""

    base = Path(args.base)
    stage = str(args.stage)
    if stage not in STAGES:
        raise GuardRefusal("STAGE_UNKNOWN", stage)
    check_not_production(base)
    content, digest = prereg.load(base)
    model = model_from_prereg(content)
    # a command-line value can only LOWER a cap: ``Caps`` refuses one above the Human's limit
    caps = Caps(max_subjects=args.max_subjects or HARD_MAX_SUBJECTS, turn_seconds=args.turn_seconds or HARD_TURN_SECONDS)
    preflight(base, caps, stage, digest, content, ledger_path)
    with lock or RealLock(base):
        assert model.revision is not None
        launch = attest(stage, digest, human_flag=bool(args.human_launch), revision8=model.revision[:8])
        revision = check_pinned_revision(model)  # the cache holds exactly the pinned digest
        reading = gpu.check_free(caps.min_free_bytes, gpu_runner)  # CUDA runtime, never NVML
        artifacts.mark_base(base, digest)
        launch_id = uuid.uuid4().hex[:10]
        budget = Session(Ledger(ledger_path), caps, launch_id, stage)
        stopper = Stopper(base, budget)
        stopper.install()
        ctx = RunContext(
            base=base, approver=HumanApprover(print, args.ssh_host), model=model, launch=launch, budget=budget,
            turn_cap_s=caps.turn_seconds, prereg_sha256=digest,
            gpu_check=lambda: gpu.check_free(caps.min_free_bytes, gpu_runner), stop_check=stopper.check,
        )
        print(f"pre-registration {digest}\nstage {stage}  model {model.model_id}  revision {revision}\n"
              f"GPU free at launch {reading.free_bytes / gpu.GIB:.1f} GiB of {reading.total_bytes / gpu.GIB:.1f} GiB\n"
              f"caps {json.dumps(caps.record(), sort_keys=True)}\nstop: python -m experiments.r20.real_run stop --base {base}  (or Ctrl-C)")
        reason = "crashed"  # only a stage that really finished is "completed"
        payload: dict[str, Any] = {}
        try:
            payload = dispatch(ctx, stage, content)
            reason = "completed"
        except KeyboardInterrupt:
            reason = stopper.reason or "interrupted"
        except GuardRefusal as refusal:
            reason = refusal.code
            payload = {"refused": refusal.code}
        except BaseException as error:  # an unexpected failure is recorded, and the results are still written
            reason = f"crashed:{type(error).__name__}"
            payload = {"exception": type(error).__name__}
        finally:
            stopper.stop_children()
            stopper.restore()
            budget.end(reason)
        path = write_results(base, launch_id, stage, {**payload, "stop_reason": reason}, digest)
        print(f"stage {stage} ended: {reason}\nresults: {path}\nreport: python -m experiments.r20.real_run report --base {base}")
        return 0 if reason == "completed" else 1


def dispatch(ctx: RunContext, stage: str, content: dict[str, Any]) -> dict[str, Any]:
    if stage == "G1":
        return {"g1": drivers.run_g1(ctx)}
    if stage == "S0":
        return {"s0": drivers.run_s0_block(ctx, kill_point=content["s0"]["kill_point"],
                                           max_attempts=int(content["s0"]["attempts_max"]))}
    if stage == "S1":
        return {"s1": drivers.run_s1_block(ctx, max_attempts=int(content["s1"]["attempts_max"]))}
    if stage == "F1":
        plan = content["f1"]
        return {"f1": drivers.run_f1(ctx, plan, lambda _pair: {})}
    raise GuardRefusal("STAGE_UNKNOWN", stage)


def _release_stage_b(args: argparse.Namespace, *, attest: Attest) -> Path:
    """The Human's release of stage B: attested like a launch, bound to the pre-registration."""

    base = Path(args.base)
    content, digest = prereg.load(base)
    attest("RELEASE-STAGE-B", digest, human_flag=bool(args.human_launch), revision8=str(content["model"]["revision_pinned"])[:8])
    path = Path(base) / drivers.STAGE_B_RELEASE
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"prereg_sha256": digest, "released_by": "human", "at": time.time()}, handle)
    return path


def run_stage(args: argparse.Namespace) -> int:
    """The public entry: NO seam (real terminal, real environment, real ancestry, real GPU probe,
    the per-user cumulative ledger)."""

    return _run_stage(args, attest=attest_human_launch, gpu_runner=None, lock=None, ledger_path=default_ledger_path())


def release_stage_b(args: argparse.Namespace) -> Path:
    return _release_stage_b(args, attest=attest_human_launch)


def _create_prereg(args: argparse.Namespace, *, attest: Attest) -> str:
    """The pre-registration is the Human's act: ``--human-launch``, a terminal, the four known
    violations typed at the terminal, the whole content shown, and a confirmation sentence that
    carries the stage and the first 8 characters of the model revision."""

    base = Path(args.base)
    check_not_production(base)  # BEFORE anything is created
    head = factory.git_head()
    if head == "unknown":
        raise GuardRefusal("GIT_HEAD_UNKNOWN", "the git head cannot be read")
    if factory.git_dirty() is not False:
        raise GuardRefusal("GIT_TREE_DIRTY", "pre-register from a clean checkout")
    model = ModelOverride(
        provider="transformers", model_id=args.model_id, load_in_4bit=not args.no_4bit, max_new_tokens=args.max_new_tokens,
        hf_home=Path(args.hf_home), revision=args.model_revision,
    )
    assert model.revision is not None
    typed: list[str] = []

    def gather(terminal: Any) -> list[str]:
        for index in range(1, 5):
            typed.append(terminal.prompt(f"Known violation {index} of 4 (the Human's own words, one line)").strip())
        prereg.check_four(typed)
        return [
            f"model id: {model.model_id}", f"hf_home: {model.hf_home}", f"revision: {model.revision}",
            f"git head: {head}", f"caps: {json.dumps(Caps().record(), sort_keys=True)}",
            *[f"known violation {i}: {text}" for i, text in enumerate(typed, 1)],
        ]

    attest("PREREG", "NEW", human_flag=bool(args.human_launch), revision8=model.revision[:8], before_confirm=gather)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    content = prereg.build_content(
        git_head=head, model=model, caps=Caps(), four_violations=typed, master_seed=args.master_seed, trial_id=args.trial_id,
    )
    return prereg.create(base, content)


def create_prereg(args: argparse.Namespace) -> str:
    return _create_prereg(args, attest=attest_human_launch)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="experiments.r20.real_run")
    commands = top.add_subparsers(dest="command", required=True)
    one = commands.add_parser("prereg")
    one.add_argument("--base", required=True)
    one.add_argument("--model-id", required=True, help="explicit org/name; never read from a host")
    one.add_argument("--hf-home", required=True, help="absolute HF cache directory (read only)")
    one.add_argument("--model-revision", required=True, help="the pinned 40-hex commit digest")
    one.add_argument("--no-4bit", action="store_true")
    one.add_argument("--max-new-tokens", type=int)
    one.add_argument("--master-seed", type=int, required=True)
    one.add_argument("--trial-id", required=True)
    one.add_argument("--human-launch", action="store_true")
    two = commands.add_parser("run")
    two.add_argument("--base", required=True)
    two.add_argument("--stage", required=True, choices=STAGES)
    two.add_argument("--human-launch", action="store_true")
    two.add_argument("--ssh-host", default=os.uname().nodename, help="the ssh name of this host, for the printed tunnel line")
    two.add_argument("--max-subjects", type=int, help="can only LOWER the cap of 42")
    two.add_argument("--turn-seconds", type=float, help="can only LOWER the cap of 120")
    three = commands.add_parser("release-stage-b")
    three.add_argument("--base", required=True)
    three.add_argument("--human-launch", action="store_true")
    four = commands.add_parser("stop")
    four.add_argument("--base", required=True)
    five = commands.add_parser("report")
    five.add_argument("--base", required=True)
    five.add_argument("--out")
    six = commands.add_parser("erase")
    six.add_argument("--base", required=True)
    six.add_argument("--run-id", action="append")
    six.add_argument("--older-than-days", type=float)
    six.add_argument("--confirm-erase", action="store_true")
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prereg":
            print(create_prereg(args))
        elif args.command == "run":
            return run_stage(args)
        elif args.command == "release-stage-b":
            print(release_stage_b(args))
        elif args.command == "stop":
            (Path(args.base) / STOP_FILE).write_text("stop\n", encoding="utf-8")
            print("stop file written; the launcher stops its process group within a second")
        elif args.command == "report":
            print(write_report(Path(args.base), Path(args.out) if args.out else None))
        elif args.command == "erase":
            for item in artifacts.erase(Path(args.base), run_ids=args.run_id, older_than_days=args.older_than_days,
                                        confirmed=bool(args.confirm_erase)):
                print(f"erased {item.run_id}: {', '.join(item.removed)}")
    except GuardRefusal as refusal:
        print(f"real_run refused: {refusal.code} {refusal.detail}".rstrip(), file=sys.stderr)
        return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["RealLock", "Stopper", "main", "model_from_prereg", "preflight", "run_stage", "ssh_line"]
