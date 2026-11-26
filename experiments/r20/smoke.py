"""A lightweight real-model smoke (conversation only): ``python -m experiments.r20.smoke``.

The Human's command (on the lab host, in a real terminal; nothing here may be run by an agent):

    python -m experiments.r20.smoke --base B --model-id ORG/NAME --hf-home /abs/hf \
        --model-revision <40-hex> --human-launch [--no-4bit] [--max-new-tokens N] [--turns K] \
        [--temperature T] [--repeats N] [--record-model-calls] [--ssh-host H]

What it does: refuses a production base, checks the pinned model revision in the explicit HF cache,
asks the Human to type the launch sentence (``launch.attest_human_launch``), checks that the GPU has
room, creates ONE fresh throwaway subject (never production state), activates it, holds up to K
(at most 7) fixed neutral Japanese chat turns and REPORTS WHAT IT OBSERVED.

Diagnostic options (all observation only, all default off / unchanged):

* ``--temperature T`` (0.0..1.5; 0 means greedy, ``do_sample`` false) overrides the lab subject's
  generation config (``ModelOverride``); the requested and the effective values are in the results.
  Without it the config's own value is used.
* ``--repeats N`` (1..5) runs N sequential FRESH subjects with the same turn script under ONE Human
  attestation.  The GPU free check runs before every subject; the loop stops at the first refusal
  or non-completed run and the completed runs are reported.  The results file lists every run and a
  table of counts / means over them.
* ``--conversation-trace`` sets ``conversation_trace.enabled: true`` in the lab subject's generated config
  (every other key of that section stays the repository default; the one-way AgentState V10 -> V11 move
  happens on a throwaway genesis subject only).  The repository config has it off, so without the flag
  the model input holds no earlier turn.  Results carry ``conversation_trace: {requested, config_applied}``
  (``config_applied`` is read back from the subject's own config file) beside the prompt-assembly numbers.
* ``--chat-roles`` (needs ``--conversation-trace`` and ``--record-model-calls``) installs the lab-only
  ``chat_role_adapter`` in the subject's serve child: the recent conversation is sent to the model as real
  alternating chat roles (system, user / assistant pairs, final user) instead of text inside ONE user message.
  Results carry ``chat_roles: {requested, applied, ...}`` and, per turn, numbers only (pairs moved, user
  characters before / after, skipped-reason labels); the model text and the messages sent stay private.
* ``--record-model-calls`` installs the lab-only ``recording_provider`` in the subject's serve
  child: the exact prompt / messages, the sampling parameters and the output of every model call go
  to the subject's PRIVATE directory (0600).  The results then carry, per turn, NUMBERS about how
  the production chat path assembled the model input (message count, prior turns present as text,
  prompt characters), never the text.  Without the flag no wrapper is installed and no file is made.

What it is NOT: a test of the model.  There is no pass / fail on the model's behaviour, no reply
normalisation, no regeneration or retry, no fixed target, no pre-registration, no cap ledger and no
stop file.  A model that ignores an instruction is an observation, not a failure.  Every label is
``kind: observed``.  The exit code is 0 whenever the run itself completed; it is non-zero only for a
guard / infrastructure refusal (2) or an infrastructure failure (1: a turn timeout, a dead child, an
unexpected exception).

Outputs: ``<base>/results-smoke-<launch>.json`` (numbers and labels only, NO model text) and the raw
replies in each subject's private directory (0700 directory, 0600 file, with a retention note).

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

import yaml

from experiments.r15_p0.pilot import Case
from experiments.r20 import artifacts, chat_role_adapter, gpu, measure
from experiments.r20.guards import GuardRefusal, check_not_production
from experiments.r20.launch import HumanLaunch, Terminal, attest_human_launch
from experiments.r20.model_config import MAX_TEMPERATURE, ModelOverride, check_pinned_revision, model_digest
from experiments.r20.recording_provider import assembly_numbers, model_input_text, read_calls
from experiments.r20.s0 import S0Session
from suzka.r15.qualified_proposal import prior_referents

STAGE = "SMOKE"
TURN_CAP_S = 120.0  # one chat turn (the per-turn timeout of the child and of the client)
MAX_TURNS = 7
MAX_REPEATS = 5
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
    ("population", "では、その都市の人口はおよそ何人ですか？"),
    ("recall", "最初にお伝えした案Aと案Bの内容を、覚えている範囲で教えてください。"),
)
# The option phrases registered by the setup prompt (the recall turn is observed against them).
OPTION_PHRASES = ("初期費用を抑える", "保守費用を抑える")
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


def observe_reply(label: str, response: str) -> dict[str, Any] | None:
    """Plain substring OBSERVATIONS for the follow-up turns (never a verdict; ``None`` for other labels).

    recall: does the reply contain the exact option phrases the setup prompt registered?
    rephrase: does it mention 東京 (the immediately preceding answer) / 案 or 比較 (the earlier comparison)?
    population: does it mention 東京 / 都市?
    """

    if label == "recall":
        first, second = (phrase in response for phrase in OPTION_PHRASES)
        return {"kind": "observed", "has_initial_cost_phrase": first, "has_maintenance_cost_phrase": second,
                "has_both_phrases": first and second}
    if label == "rephrase":
        return {"kind": "observed", "mentions_tokyo": "東京" in response,
                "mentions_plan_or_comparison": "案" in response or "比較" in response}
    if label == "population":
        return {"kind": "observed", "mentions_tokyo": "東京" in response, "mentions_city_word": "都市" in response}
    return None


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and means over the runs (numbers only).  Booleans are counted, other numbers averaged."""

    def fold(rows: list[dict[str, Any]]) -> dict[str, Any]:
        table: dict[str, Any] = {"n": len(rows)}
        for name in sorted({key for row in rows for key in row if key not in {"kind", "method"}}):
            values = [row[name] for row in rows if row.get(name) is not None]
            if values and all(isinstance(value, bool) for value in values):
                table[name] = {"true": sum(values), "of": len(values)}
            elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                table[name] = {"mean": round(sum(values) / len(values), 3), "of": len(values)}
        return table

    def fold_roles(rows: list[dict[str, Any]]) -> dict[str, Any]:
        table = fold([{key: value for key, value in row.items() if key != "adapter_skipped"} for row in rows])
        reasons = sorted({row["adapter_skipped"] for row in rows if row.get("adapter_skipped")})
        table["adapter_skipped"] = {reason: sum(1 for row in rows if row.get("adapter_skipped") == reason) for reason in reasons}
        return table

    labels = [label for label, _text in TURNS]
    per_label: dict[str, Any] = {}
    for label in labels:
        turns = [turn for run in runs for turn in run["turns"] if turn["label"] == label]
        if not turns:
            continue
        observed = [turn["option_naming"] or turn["observed_labels"] for turn in turns]
        assembly = [turn["prompt_assembly"] for turn in turns if turn["prompt_assembly"] is not None]
        roles = [turn["chat_roles"] for turn in turns if turn.get("chat_roles") is not None]
        per_label[label] = {
            "turns": len(turns),
            "mean_seconds": round(sum(turn["seconds"] for turn in turns) / len(turns), 3),
            "mean_response_chars": round(sum(turn["response_chars"] for turn in turns) / len(turns), 3),
            "observed": fold([row for row in observed if row is not None]) if any(observed) else None,
            "prompt_assembly": fold(assembly) if assembly else None,
            "chat_roles": fold_roles(roles) if roles else None,
        }
    return {"kind": "observed", "runs": len(runs), "per_label": per_label}


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
        f"temperature: {'config value' if model.temperature is None else model.temperature}",
        f"conversation trace enabled in the subject config: {bool(args.conversation_trace)}",
        f"recent conversation sent as chat roles (lab adapter): {bool(getattr(args, 'chat_roles', False))}",
        f"fresh subjects (repeats): {args.repeats}; model calls recorded privately: {bool(args.record_model_calls)}",
        "a fresh throwaway subject is created per repeat; no production state is touched; observations only",
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
        temperature=args.temperature,
    )


def _overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    return {"conversation_trace.enabled": True} if args.conversation_trace else None


def _trace_record(args: argparse.Namespace, config_path: Path | None) -> dict[str, bool]:
    """Labels only: was it asked for, and does the subject's OWN config file say it is enabled?"""

    applied = False
    if config_path is not None:
        try:
            section = (yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}).get("conversation_trace")
            applied = isinstance(section, dict) and section.get("enabled") is True
        except (OSError, yaml.YAMLError):
            applied = False
    return {"requested": bool(args.conversation_trace), "config_applied": applied}


def _roles_record(args: argparse.Namespace, turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Labels and counts only.  `applied`: the adapter was in the model path (it reported on a call)."""

    seen = [turn["chat_roles"] for turn in turns if turn.get("chat_roles") is not None]
    return {
        "requested": bool(getattr(args, "chat_roles", False)),
        "applied": bool(seen),
        "turns_restructured": sum(1 for row in seen if row.get("applied") is True),
        "turns_passed_through": sum(1 for row in seen if row.get("applied") is not True),
    }


def _check_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.turns <= MAX_TURNS:
        raise GuardRefusal("TURNS_OUT_OF_RANGE", f"1..{MAX_TURNS}")
    if not 1 <= args.repeats <= MAX_REPEATS:
        raise GuardRefusal("REPEATS_OUT_OF_RANGE", f"1..{MAX_REPEATS}")
    if args.temperature is not None and not 0.0 <= args.temperature <= MAX_TEMPERATURE:  # NaN fails too
        raise GuardRefusal("TEMPERATURE_OUT_OF_RANGE", f"0.0..{MAX_TEMPERATURE}")
    if getattr(args, "chat_roles", False):
        if not args.conversation_trace:
            raise GuardRefusal("CHAT_ROLES_NEEDS_CONVERSATION_TRACE", "--chat-roles needs --conversation-trace")
        if not args.record_model_calls:
            raise GuardRefusal("CHAT_ROLES_NEEDS_RECORD_MODEL_CALLS", "--chat-roles needs --record-model-calls")


def _one_run(
    args: argparse.Namespace, base: Path, launch_id: str, ordinal: int, *, model: ModelOverride | None,
    launch: HumanLaunch | None, reading: gpu.GpuReading | None, revision: str,
) -> tuple[dict[str, Any], int]:
    """ONE fresh subject, the whole turn script.  Returns (the run's numbers and labels, exit code)."""

    session = S0Session(
        base, profile="S0", model=model, launch=launch, turn_cap_s=TURN_CAP_S,
        script=_SCRIPTED if model is None else None, default_answer=_SCRIPTED_DEFAULT if model is None else None,
        record_calls=bool(args.record_model_calls),
        chat_roles=bool(getattr(args, "chat_roles", False)),
        config_overrides=_overrides(args),
    )
    fixture = model is None
    label = "FIXTURE ONLY (scripted provider, no model, no GPU)" if fixture else f"REAL MODEL {model.model_id if model else ''}"
    print(f"smoke {launch_id} run {ordinal}/{args.repeats}: {label}\n"
          f"subject {session.subject.run_id} (throwaway, port {session.subject.port})")
    if args.ssh_host:
        print(f"to look at the subject: ssh -N -L {session.subject.port}:127.0.0.1:{session.subject.port} {args.ssh_host}")
    private_directory = artifacts.private_dir(session.subject.run_dir)
    private_file = private_directory / f"smoke-{launch_id}.jsonl"
    artifacts.write_private(private_file, b"")
    turns: list[dict[str, Any]] = []
    ended = "completed"
    detail = ""
    code = 0
    previous: tuple[str, str] | None = None  # (request, response) of the immediately preceding turn
    try:
        session.start()
        session.activate()
        for turn_ordinal, (turn_label, text) in enumerate(TURNS[: args.turns], start=1):
            seen = len(read_calls(private_directory)) if args.record_model_calls else 0
            reply = session.chat(text)
            response = str(reply.get("response", ""))
            _keep(private_file, turn_ordinal, turn_label, text, response)
            calls: int | None = None
            sampling: dict[str, Any] | None = None
            assembly: dict[str, Any] | None = None
            roles: dict[str, Any] | None = None
            if args.record_model_calls:
                rows = read_calls(private_directory)[seen:]
                calls = len(rows)
                # the call that carried THIS turn's message (the last one whose input holds it)
                carrier = next((row for row in reversed(rows) if text in model_input_text(row)), None)
                if carrier is not None:
                    sampling = carrier.get("sampling")
                    roles = chat_role_adapter.numbers(carrier)
                    assembly = assembly_numbers(
                        carrier,
                        previous_request=None if previous is None else previous[0],
                        previous_response=None if previous is None else previous[1],
                    )
            previous = (text, response)
            turns.append({
                "ordinal": turn_ordinal, "label": turn_label, "seconds": session.turn_seconds[-1],
                "response_chars": len(response), "option_naming": observe_option_naming(turn_label, response),
                "observed_labels": observe_reply(turn_label, response),
                "model_calls": calls, "sampling": sampling, "prompt_assembly": assembly, "chat_roles": roles,
            })
    except GuardRefusal as refusal:  # a guard / infrastructure refusal (a turn timeout is one)
        ended, detail, code = refusal.code, refusal.detail, EXIT_REFUSED
    except BaseException as error:  # noqa: BLE001 - recorded, results still written, child still stopped
        ended, code = f"crashed:{type(error).__name__}", EXIT_FAILED
    finally:
        session.stop()
    gpu_rows = measure.read_log(session.subject.evidence_dir / "gpu.jsonl")
    run: dict[str, Any] = {
        "run": ordinal,
        "run_id": session.subject.run_id,
        "provider": session.provider_identity(),
        "model_revision": revision or None,
        "ended": ended,
        "serving_after_s": None if session.serving_after_s is None else round(session.serving_after_s, 3),
        "gpu": {"free_at_launch_bytes": None if reading is None else reading.free_bytes, **measure.summarize(gpu_rows)},
        "conversation_trace": _trace_record(args, session.subject.config_path),
        "chat_roles": _roles_record(args, turns),
        "turns": turns,
        "private_file": str(private_file),
        "model_calls_file": str(private_directory / "model_calls.jsonl") if args.record_model_calls else None,
    }
    if detail:
        run["detail"] = detail
    return run, code


def _print_run(run: dict[str, Any], reading: gpu.GpuReading | None) -> None:
    print(f"ended: {run['ended']}" + (f" ({run['detail']})" if run.get("detail") else ""))
    if run["serving_after_s"] is not None:
        print(f"model load / serving after: {run['serving_after_s']:.1f} s")
    if reading is not None:
        summary = run["gpu"]
        print(f"GPU free at launch: {reading.free_bytes / gpu.GIB:.1f} GiB of {reading.total_bytes / gpu.GIB:.1f} GiB"
              f"; peak reserved: {int(summary.get('peak_reserved', 0)) / gpu.GIB:.1f} GiB")
    for item in run["turns"]:
        naming = item["option_naming"]
        note = "" if naming is None else f"  observed: options named {naming['options_named']}"
        extra = item["observed_labels"]
        if extra is not None:
            note += "  observed: " + ", ".join(f"{key}={value}" for key, value in extra.items() if key != "kind")
        assembly = item["prompt_assembly"]
        if assembly is not None:
            roles = item.get("chat_roles")
            if roles is not None:
                note += f"  chat roles: pairs moved {roles['pairs_moved']}, skipped {roles['adapter_skipped']}"
            note += (f"  input: {assembly['messages']} msg, {assembly['prompt_chars']} chars, "
                     f"trace turns {assembly['trace_turns']}, episodic items {assembly['episodic_items']}")
        print(f"turn {item['ordinal']} [{item['label']}]: {item['seconds']:.1f} s, {item['response_chars']} chars{note}")
    print(f"raw replies (PRIVATE): {run['private_file']}")


def _smoke(args: argparse.Namespace, *, attest: Attest, gpu_runner: Callable[[], str] | None) -> int:
    """The smoke.  PRIVATE seam version (``attest`` / ``gpu_runner``): only tests pass anything else;
    ``run`` below is hard-wired to the real terminal and the real GPU probe."""

    _check_arguments(args)
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
    fixture = model is None
    runs: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None
    code = 0
    for ordinal in range(1, args.repeats + 1):
        if ordinal > 1 and model is not None:
            try:
                reading = gpu.check_free(runner=gpu_runner)  # before EVERY subject
            except GuardRefusal as refusal:
                stopped = {"before_run": ordinal, "code": refusal.code, "detail": refusal.detail}
                code = EXIT_REFUSED
                print(f"stopped before run {ordinal}: {refusal.code} {refusal.detail}".rstrip())
                break
        run, run_code = _one_run(
            args, base, launch_id, ordinal, model=model, launch=launch, reading=reading, revision=revision,
        )
        runs.append(run)
        _print_run(run, reading)
        if run_code:  # a non-completed run: do not start another subject
            code = run_code
            if ordinal < args.repeats:
                stopped = {"before_run": ordinal + 1, "code": run["ended"], "detail": "the previous run did not complete"}
            break
    completed = [run for run in runs if run["ended"] == "completed"]
    payload: dict[str, Any] = {
        "kind": "observations",
        "verdict": None,  # never a pass / fail on the model's behaviour
        "fixture_only": fixture,
        "launch": launch_id,
        "requested": {"repeats": args.repeats, "turns": args.turns, "record_model_calls": bool(args.record_model_calls)},
        "generation": {
            "temperature_requested": args.temperature,
            "temperature_zero_means_greedy": True,
            "override_applied": model is not None and args.temperature is not None,
            "do_sample_override": None if args.temperature is None else args.temperature != 0.0,
            "config_value_kept": args.temperature is None,
        },
        "conversation_trace": {
            "requested": bool(args.conversation_trace),
            "config_applied": bool(runs) and all(run.get("conversation_trace", {}).get("config_applied") is True for run in runs),
        },
        "chat_roles": {
            "requested": bool(getattr(args, "chat_roles", False)),
            "applied": bool(completed) and all(run.get("chat_roles", {}).get("applied") is True for run in completed),
            "turns_restructured": sum(run.get("chat_roles", {}).get("turns_restructured", 0) for run in runs),
            "turns_passed_through": sum(run.get("chat_roles", {}).get("turns_passed_through", 0) for run in runs),
        },
        "runs": runs,
        "completed_runs": len(completed),
        "stopped": stopped,
        "aggregate": aggregate(completed),
    }
    results = _write_results(base, launch_id, payload)
    print(f"completed runs: {len(completed)} of {args.repeats}" + (f"; stopped: {stopped['code']}" if stopped else ""))
    print(f"results (numbers and labels only): {results}")
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
    top.add_argument("--temperature", type=float,
                     help=f"override generation.temperature, 0.0..{MAX_TEMPERATURE} (0 = greedy); default: the config value")
    top.add_argument("--repeats", type=int, default=1, help=f"sequential fresh subjects under one attestation, 1..{MAX_REPEATS}")
    top.add_argument("--record-model-calls", action="store_true",
                     help="lab diagnostics: record every model call privately and report prompt-assembly numbers")
    top.add_argument("--conversation-trace", action="store_true",
                     help="lab diagnostics: enable conversation_trace in the throwaway subject's config (default off)")
    top.add_argument("--chat-roles", action="store_true",
                     help="lab diagnostics: send the recent conversation as real chat roles (needs --conversation-trace and --record-model-calls)")
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


__all__ = ["OPTION_PHRASES", "TURNS", "aggregate", "main", "observe_option_naming", "observe_reply", "run"]
