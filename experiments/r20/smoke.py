"""A lightweight real-model smoke (conversation only): ``python -m experiments.r20.smoke``.

The Human's command (on the lab host, in a real terminal; nothing here may be run by an agent):

    python -m experiments.r20.smoke --base B --model-id ORG/NAME --hf-home /abs/hf \
        --model-revision <40-hex> --human-launch [--no-4bit] [--max-new-tokens N] [--turns K] \
        [--ssh-host H]

What it does: refuses a production base, checks the pinned model revision in the explicit HF cache,
asks the Human to type the launch sentence (``launch.attest_human_launch``), checks that the GPU has
room, creates ONE fresh throwaway subject (never production state), activates it, holds up to K
(at most 6) fixed neutral Japanese chat turns and REPORTS WHAT IT OBSERVED.

What it is NOT: a test of the model.  There is no pass / fail on the model's behaviour, no reply
normalisation, no regeneration or retry, no fixed target, no pre-registration, no cap ledger and no
stop file.  A model that ignores an instruction is an observation, not a failure.  The exit code is
0 whenever the run itself completed; it is non-zero only for a guard / infrastructure refusal (2) or
an infrastructure failure (1: a turn timeout, a dead child, an unexpected exception).

Outputs: ``<base>/results-smoke-<launch>.json`` (numbers and labels only, NO model text) and the raw
replies in the subject's private directory (0700 directory, 0600 file, with a retention note).

``--scripted`` runs the same flow with the scripted provider, no GPU and no Human launch.  It exists
for the tests and is labelled FIXTURE ONLY in every output; it says nothing about a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid

from experiments.r15_p0.pilot import Case
from experiments.r20 import artifacts, gpu, measure
from experiments.r20.guards import GuardRefusal, check_not_production
from experiments.r20.launch import HumanLaunch, Terminal, attest_human_launch
from experiments.r20.model_config import ModelOverride, check_pinned_revision, model_digest
from experiments.r20.s0 import S0Session
from suzka.r15.qualified_proposal import prior_referents

STAGE = "SMOKE"
TURN_CAP_S = 120.0  # one chat turn (the per-turn timeout of the child and of the client)
MAX_TURNS = 6
EXIT_REFUSED = 2
EXIT_FAILED = 1
Attest = Callable[..., HumanLaunch]

_CASE = Case("r20-smoke", "B", "forced", 0, "normal")  # only its prompt texts are used; no target is ever applied
_PROMPTS = _CASE.prompts()
# (label, text).  setup + E ask the model to pick A or B: a plain observation of instruction following.
TURNS: tuple[tuple[str, str], ...] = (
    ("setup", _PROMPTS["setup"]),
    ("E", _PROMPTS["E"]),
    ("identity", "あなたは誰ですか？短く答えてください。"),
    ("factual", "日本の首都はどこですか？"),
    ("rephrase", "今の答えを、もう一度一文で言い換えてください。"),
    ("recall", "最初にお伝えした案Aと案Bの内容を、覚えている範囲で教えてください。"),
)
# Scripted fixture answers by call index: the E answer names TWO options on purpose (an observation, not a failure).
_SCRIPTED = (
    "synthetic fixture answer",
    "案Aを暫定的に選びましたが、案Bを暫定的に選ぶこともできます。",
)
_SCRIPTED_DEFAULT = "synthetic fixture answer"


def observe_option_naming(label: str, response: str) -> dict[str, Any] | None:
    """An OBSERVATION of how many options the E reply names (never a verdict).  Only the E turn asks."""

    if label != "E":
        return None
    named = sorted(prior_referents(response))
    return {"kind": "observed", "options_named": len(named), "names_exactly_one": len(named) == 1}


def _write_results(base: Path, launch_id: str, payload: dict[str, Any]) -> Path:
    path = Path(base) / f"results-smoke-{launch_id}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=1, default=str)
    return path


def _keep(path: Path, ordinal: int, label: str, request: str, response: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            {"ordinal": ordinal, "label": label, "request": request, "response": response}, ensure_ascii=False,
        ) + "\n")


def _summary_lines(args: argparse.Namespace, model: ModelOverride | None) -> list[str]:
    assert model is not None
    return [
        f"model id: {model.model_id}", f"revision: {model.revision}", f"hf_home: {model.hf_home}",
        f"4-bit: {model.load_in_4bit}", f"base: {args.base}", f"chat turns: up to {args.turns}",
        "a fresh throwaway subject is created; no production state is touched; observations only",
    ]


def _model(args: argparse.Namespace) -> ModelOverride | None:
    given = [args.model_id, args.hf_home, args.model_revision]
    if args.scripted:
        if any(given):
            raise GuardRefusal("SCRIPTED_WITH_MODEL_ARGUMENTS", "--scripted takes no model arguments")
        return None
    if not all(given):
        raise GuardRefusal("MODEL_ARGUMENTS_REQUIRED", "--model-id, --hf-home and --model-revision")
    return ModelOverride(
        provider="transformers", model_id=args.model_id, load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens, hf_home=Path(args.hf_home), revision=args.model_revision,
    )


def _smoke(args: argparse.Namespace, *, attest: Attest, gpu_runner: Callable[[], str] | None) -> int:
    """The smoke.  PRIVATE seam version (``attest`` / ``gpu_runner``): only tests pass anything else;
    ``run`` below is hard-wired to the real terminal and the real GPU probe."""

    if not 1 <= args.turns <= MAX_TURNS:
        raise GuardRefusal("TURNS_OUT_OF_RANGE", f"1..{MAX_TURNS}")
    base = Path(args.base)
    check_not_production(base)  # BEFORE anything is created
    model = _model(args)
    launch: HumanLaunch | None = None
    reading: gpu.GpuReading | None = None
    if model is not None:
        revision = check_pinned_revision(model)  # the cache holds exactly the pinned digest
        assert model.revision is not None
        def before(_terminal: Terminal) -> list[str]:
            return _summary_lines(args, model)

        launch = attest(
            STAGE, model_digest(model), human_flag=bool(args.human_launch), revision8=model.revision[:8],
            before_confirm=before,
        )
        reading = gpu.check_free(runner=gpu_runner)  # CUDA runtime, never NVML; named minimum in gpu.py
    else:
        revision = ""
    launch_id = uuid.uuid4().hex[:10]
    session = S0Session(
        base, profile="S0", model=model, launch=launch, turn_cap_s=TURN_CAP_S,
        script=_SCRIPTED if model is None else None, default_answer=_SCRIPTED_DEFAULT if model is None else None,
    )
    fixture = model is None
    label = "FIXTURE ONLY (scripted provider, no model, no GPU)" if fixture else f"REAL MODEL {model.model_id if model else ''}"
    print(f"smoke {launch_id}: {label}\nsubject {session.subject.run_id} (throwaway, port {session.subject.port})")
    if args.ssh_host:
        print(f"to look at the subject: ssh -N -L {session.subject.port}:127.0.0.1:{session.subject.port} {args.ssh_host}")
    private_file = artifacts.private_dir(session.subject.run_dir) / f"smoke-{launch_id}.jsonl"
    artifacts.write_private(private_file, b"")
    turns: list[dict[str, Any]] = []
    ended = "completed"
    detail = ""
    code = 0
    try:
        session.start()
        session.activate()
        for ordinal, (turn_label, text) in enumerate(TURNS[: args.turns], start=1):
            reply = session.chat(text)
            response = str(reply.get("response", ""))
            _keep(private_file, ordinal, turn_label, text, response)
            turns.append({
                "ordinal": ordinal, "label": turn_label, "seconds": session.turn_seconds[-1],
                "response_chars": len(response), "option_naming": observe_option_naming(turn_label, response),
            })
    except GuardRefusal as refusal:  # a guard / infrastructure refusal (a turn timeout is one)
        ended, detail, code = refusal.code, refusal.detail, EXIT_REFUSED
    except BaseException as error:  # noqa: BLE001 - recorded, results still written, child still stopped
        ended, code = f"crashed:{type(error).__name__}", EXIT_FAILED
    finally:
        session.stop()
    gpu_rows = measure.read_log(session.subject.evidence_dir / "gpu.jsonl")
    payload: dict[str, Any] = {
        "kind": "observations",
        "verdict": None,  # never a pass / fail on the model's behaviour
        "fixture_only": fixture,
        "launch": launch_id,
        "run_id": session.subject.run_id,
        "provider": session.provider_identity(),
        "model_revision": revision or None,
        "ended": ended,
        "serving_after_s": None if session.serving_after_s is None else round(session.serving_after_s, 3),
        "gpu": {"free_at_launch_bytes": None if reading is None else reading.free_bytes, **measure.summarize(gpu_rows)},
        "turns": turns,
        "private_file": str(private_file),
    }
    if detail:
        payload["detail"] = detail
    results = _write_results(base, launch_id, payload)
    print(f"ended: {ended}" + (f" ({detail})" if detail else ""))
    if session.serving_after_s is not None:
        print(f"model load / serving after: {session.serving_after_s:.1f} s")
    if reading is not None:
        summary = payload["gpu"]
        print(f"GPU free at launch: {reading.free_bytes / gpu.GIB:.1f} GiB of {reading.total_bytes / gpu.GIB:.1f} GiB"
              f"; peak reserved: {int(summary.get('peak_reserved', 0)) / gpu.GIB:.1f} GiB")  # type: ignore[call-overload]
    for item in turns:
        naming = item["option_naming"]
        note = "" if naming is None else f"  observed: options named {naming['options_named']}"
        print(f"turn {item['ordinal']} [{item['label']}]: {item['seconds']:.1f} s, {item['response_chars']} chars{note}")
    print(f"results (numbers and labels only): {results}\nraw replies (PRIVATE): {private_file}")
    print(artifacts.RETENTION_NOTE.strip())
    print("These are observations, not a verdict on the model.")
    return code


def run(args: argparse.Namespace) -> int:
    """The public entry: real terminal, real environment, real ancestry, real GPU probe."""

    return _smoke(args, attest=attest_human_launch, gpu_runner=None)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="experiments.r20.smoke", description=__doc__.split("\n\n")[0])
    top.add_argument("--base", required=True, help="absolute directory for the throwaway subject (never production)")
    top.add_argument("--model-id", help="explicit org/name; never read from a host")
    top.add_argument("--hf-home", help="absolute HF cache directory (read only)")
    top.add_argument("--model-revision", help="the pinned 40-hex commit digest")
    top.add_argument("--no-4bit", action="store_true")
    top.add_argument("--max-new-tokens", type=int)
    top.add_argument("--turns", type=int, default=MAX_TURNS, help=f"number of chat turns, 1..{MAX_TURNS}")
    top.add_argument("--human-launch", action="store_true")
    top.add_argument("--ssh-host", help="ssh name of this host, for the printed tunnel line")
    top.add_argument("--scripted", action="store_true", help="FIXTURE ONLY: scripted provider, no GPU, no model")
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except GuardRefusal as refusal:
        print(f"smoke refused: {refusal.code} {refusal.detail}".rstrip(), file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["TURNS", "main", "observe_option_naming", "run"]
