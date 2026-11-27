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
* ``--script short|long`` (default short = the 7-turn script; ``--turns`` 1..7).  ``long`` is a fixed 16-turn script
  (``--turns`` 1..16): setup, E (turn 2), identity, nine neutral filler turns, then the probes at turns 13-16 (factual,
  rephrase, population, recall) with the same observation labels, so the first-turn state block (X9 / X10) is 12 turns
  away from them.  Every mode records per turn ``lyrical_marker_hits`` (occurrences of ``LYRICAL_MARKERS``, a lab list:
  #414 names only 5 of its words) and ``response_chars``; aggregates give their means per label and per condition,
  and the matrix table adds "lyric hits" and "mean chars".  Observations only; the raw text stays private.
  Example: ``--matrix --matrix-conditions X0,X7,X8,X9,X10 --script long --temperature 0.7 --record-model-calls``
  (25 subjects x 16 turns; ESTIMATED roughly 4-6 minutes per subject, about 2-2.5 hours: an assumption).
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
* ``--bare-chat`` (needs ``--record-model-calls``; not with ``--chat-roles``; ``--conversation-trace`` not needed)
  installs the lab-only ``bare_chat_adapter`` in the subject's serve child: a CONTROL.  The model receives only
  a fixed neutral system text, the adapter's own (user message, model output) history and the current user
  message; nothing of SUZKA's prompt (values, goals, evidence, trace).  The SUZKA runtime runs normally around it.
  Results carry ``bare_chat: {requested, applied, ...}`` and, per turn, numbers only (messages sent, user
  characters before / after, history pairs, skipped-reason labels); the messages stay in the private file.
* ``--bare-restore LIST`` (needs ``--bare-chat``; LIST is a comma list of header, context, values, wm_evidence,
  motivations_goals, emotion, or ``all``) restores those sections of the REAL prompt of each call, VERBATIM, into
  the bare chat (the isolation experiment: which section pulls the model off a simple instruction).
  ``--restore-placement system|user`` (default system): ``system`` appends them to the bare system text
  (deliberately ignoring the #414 rule that no stored text goes into the system role: this is a lab experiment
  on a throwaway subject); ``user`` prepends them inside the final user message (production-like).  The history
  holds only the plain (message, output) pairs, never the restored text.  A section that cannot be located
  exactly is recorded as ``section_missing:<name>`` for that call.  The restored text goes ONLY to the model and
  to the private ``model_calls.jsonl``; results and stdout carry the section names and character counts.
  Two further placements keep stored / model-originated text out of the system role (#414) and keep the final user
  message the user's text only: ``first-user`` puts ALL restored sections, as the real labelled runtime-data block
  (heading ``RUNTIME_DATA_HEADING``), in front of the FIRST user turn of the conversation, and ``split-safe`` sends
  only the code-generated sections (header, emotion, context) in the system text and the rest (values,
  wm_evidence, motivations_goals) in that first-user block.  The block is rebuilt from the CURRENT call's restored
  text and re-attached to the first user turn on every call; the stored history pairs stay raw.  Per turn the
  results add ``restored_chars_system`` / ``restored_chars_first_user`` (numbers only) for these placements.
  ``--history-window N`` / ``--state-block-max-bytes B`` (lab input caps, need ``--bare-chat``; B needs
  ``--bare-restore``) keep only the most recent N exchanges of the adapter's own history and drop whole restored sections
  from the END until the state block fits B UTF-8 bytes; the user text and assistant text are never truncated.  In the
  matrix a condition id takes the suffixes ``:w<N>`` and/or ``:b<bytes>`` (``X10:w4:b2048``); the global flags are refused
  there.  Results carry requested / effective numbers, the model-call input token count and the peak allocated VRAM.
  ``--matrix-conditions X0,X7,X9,X10`` selects (and orders) matrix conditions; X9 = all first-user, X10 = all
  split-safe exist only through it (the default matrix stays X0..X8); needs ``--matrix``; the 60-subject cap applies.
* ``--matrix`` (implies the bare chat with restores; excludes ``--bare-chat``, ``--bare-restore``,
  ``--restore-placement`` and ``--chat-roles``; needs ``--record-model-calls``; greedy unless ``--temperature``
  is given) runs ONE Human launch over the fixed conditions X0 bare, X1 header, X2 emotion, X3 values,
  X4 wm_evidence, X5 motivations_goals, X6 context, X7 all (system placement), X8 all (user placement,
  production-like), sequentially, ``--repeats`` fresh subjects each (default 5).  Total subjects are capped at
  60 (9 conditions x 5 = 45).  The GPU check runs before every subject; the loop stops at the first refusal or
  non-completed run and the completed conditions are reported.  ``results-smoke-matrix-<launch>.json`` (numbers
  and labels only) is rewritten after every subject, so Ctrl-C keeps what was done.  ESTIMATED duration (an
  assumption, not a measurement): about 3-6 minutes per subject (model load plus 7 turns), so a full matrix
  of 45 subjects takes roughly 2.5-4.5 hours.
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
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import uuid

import yaml

from experiments.r15_p0.pilot import Case
from experiments.r20 import artifacts, bare_chat_adapter, chat_role_adapter, gpu, measure
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
MATRIX_MAX_SUBJECTS = 60
MAX_CAP = 10_000_000  # sanity bound of --history-window / --state-block-max-bytes
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
# The LONG script (``--script long``): 16 turns.  Same probe semantics and the same labels as the short script, but
# the probes sit at the END (turns 13-16: factual -> rephrase -> population -> recall), at least 12 turns after the
# first-turn state block (turn 1); E stays at its early position (turn 2).  Turns 3-12 are neutral filler (a few
# invite a reflective register on purpose: the persona labels observe them); none names a city or the options.
LONG_TURNS: tuple[tuple[str, str], ...] = (
    ("setup", _PROMPTS["setup"]),
    ("E", _PROMPTS["E"]),
    ("identity", "あなたは誰ですか？短く答えてください。"),
    ("filler_season", "好きな季節とその理由を、短く教えてください。"),
    ("filler_rainy", "雨の日にできることを三つ挙げてください。"),
    ("filler_python", "Pythonでリストを逆順にする方法を一言で教えてください。"),
    ("filler_tired", "最近少し疲れているのですが、一言だけ声をかけてください。"),
    ("filler_sum", "1から10までの数の合計はいくつですか？"),
    ("filler_breakfast", "朝ごはんに向いている簡単な料理を一つ教えてください。"),
    ("filler_thanks_en", "「ありがとう」を英語で言うとどうなりますか？"),
    ("filler_week", "一週間は何日ですか？"),
    ("filler_boiling", "水は何度で沸騰しますか？"),
    ("factual", "日本の首都はどこですか？"),
    ("rephrase", "今の答えを、もう一度一文で言い換えてください。"),
    ("population", "では、その都市の人口はおよそ何人ですか？"),
    ("recall", "最初にお伝えした案Aと案Bの内容を、覚えている範囲で教えてください。"),
)
LONG_MAX_TURNS = len(LONG_TURNS)
SCRIPTS: dict[str, tuple[tuple[str, str], ...]] = {"short": TURNS, "long": LONG_TURNS}
# Persona observation (never a verdict): the marker words counted in each reply.  #414's evidence text names only
# 情緒, 穏やか, 静か, 心が, 余韻 ("..."); its full list was not recoverable, so the rest of this list is a lab-defined
# set of lyrical / poetic Japanese words in the same spirit.  A crude proxy, comparable across conditions only.
LYRICAL_MARKERS: tuple[str, ...] = (
    "情緒", "穏やか", "静か", "心が", "余韻",
    "静けさ", "しっとり", "ゆったり", "儚", "淡い", "ほのか", "風情", "面影", "揺らぎ", "思いを馳せ", "染み入", "紡", "響き",
)


def script_turns(name: str) -> tuple[tuple[str, str], ...]:
    return SCRIPTS[name]


def lyrical_marker_hits(response: str) -> int:
    """Total occurrences of the LYRICAL_MARKERS words in the reply (an observed count, never a verdict)."""

    return sum(response.count(marker) for marker in LYRICAL_MARKERS)


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
    population: does it mention 東京 / 都市, does it contain a number (a digit or 万), does it ask which city
    (都市名 / どの都市 / 指定)?  rephrase also: does it mention the capital answer (東京 or 首都)?
    """

    if label == "recall":
        first, second = (phrase in response for phrase in OPTION_PHRASES)
        return {"kind": "observed", "has_initial_cost_phrase": first, "has_maintenance_cost_phrase": second,
                "has_both_phrases": first and second}
    if label == "rephrase":
        return {"kind": "observed", "mentions_tokyo": "東京" in response,
                "mentions_plan_or_comparison": "案" in response or "比較" in response,
                "mentions_capital_answer": "東京" in response or "首都" in response}
    if label == "population":
        return {"kind": "observed", "mentions_tokyo": "東京" in response, "mentions_city_word": "都市" in response,
                "answers_with_number": any(character.isdigit() for character in response) or "万" in response,
                "asks_which_city": any(phrase in response for phrase in ("都市名", "どの都市", "指定"))}
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

    labels = list(dict.fromkeys(label for script in SCRIPTS.values() for label, _text in script))
    per_label: dict[str, Any] = {}
    for label in labels:
        turns = [turn for run in runs for turn in run["turns"] if turn["label"] == label]
        if not turns:
            continue
        observed = [turn["option_naming"] or turn["observed_labels"] for turn in turns]
        assembly = [turn["prompt_assembly"] for turn in turns if turn["prompt_assembly"] is not None]
        roles = [turn["chat_roles"] for turn in turns if turn.get("chat_roles") is not None]
        bare = [turn["bare_chat"] for turn in turns if turn.get("bare_chat") is not None]
        per_label[label] = {
            "turns": len(turns),
            "mean_seconds": round(sum(turn["seconds"] for turn in turns) / len(turns), 3),
            "mean_response_chars": round(sum(turn["response_chars"] for turn in turns) / len(turns), 3),
            "mean_lyrical_marker_hits": round(sum(turn.get("lyrical_marker_hits", 0) for turn in turns) / len(turns), 3),
            "observed": fold([row for row in observed if row is not None]) if any(observed) else None,
            "prompt_assembly": fold(assembly) if assembly else None,
            "chat_roles": fold_roles(roles) if roles else None,
            "bare_chat": fold_roles(bare) if bare else None,
        }
    every = [turn for run in runs for turn in run["turns"]]
    persona = {
        "kind": "observed", "turns": len(every),
        "mean_lyrical_marker_hits": round(sum(t.get("lyrical_marker_hits", 0) for t in every) / len(every), 3) if every else None,
        "mean_response_chars": round(sum(t["response_chars"] for t in every) / len(every), 1) if every else None,
    }
    return {"kind": "observed", "runs": len(runs), "per_label": per_label, "persona": persona}


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
        f"4-bit: {model.load_in_4bit}", f"base: {args.base}", f"chat turns: up to {args.turns} (script: {getattr(args, 'script', 'short') or 'short'})",
        f"temperature: {'config value' if model.temperature is None else model.temperature}",
        f"conversation trace enabled in the subject config: {bool(args.conversation_trace)}",
        f"recent conversation sent as chat roles (lab adapter): {bool(getattr(args, 'chat_roles', False))}",
        f"bare chat, no SUZKA prompt (lab control adapter): {bool(getattr(args, 'bare_chat', False))}",
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


def _bare_record(args: argparse.Namespace, turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Labels and counts only.  `applied`: the adapter was in the model path (it reported on a call)."""

    seen = [turn["bare_chat"] for turn in turns if turn.get("bare_chat") is not None]
    return {
        "requested": bool(getattr(args, "bare_chat", False)),
        "applied": bool(seen),
        "turns_bare": sum(1 for row in seen if row.get("applied") is True),
        "turns_passed_through": sum(1 for row in seen if row.get("applied") is not True),
    }


def _check_restore_and_matrix(args: argparse.Namespace) -> None:
    restore = getattr(args, "bare_restore", None)
    placement = getattr(args, "restore_placement", None)
    if getattr(args, "matrix", False):
        for flag, code in (("bare_chat", "MATRIX_EXCLUDES_BARE_CHAT"), ("bare_restore", "MATRIX_EXCLUDES_BARE_RESTORE"),
                           ("restore_placement", "MATRIX_EXCLUDES_RESTORE_PLACEMENT"), ("chat_roles", "MATRIX_EXCLUDES_CHAT_ROLES"),
                           ("history_window", "MATRIX_EXCLUDES_HISTORY_WINDOW"),
                           ("state_block_max_bytes", "MATRIX_EXCLUDES_STATE_BLOCK_MAX_BYTES")):
            if getattr(args, flag, None) is not None and getattr(args, flag, None) is not False:
                raise GuardRefusal(code, f"--matrix and --{flag.replace('_', '-')} are mutually exclusive")
        if not args.record_model_calls:
            raise GuardRefusal("MATRIX_NEEDS_RECORD_MODEL_CALLS", "--matrix needs --record-model-calls")
        check_conditions(resolve_conditions(getattr(args, "matrix_conditions", None)), args.repeats)  # before anything exists
    elif getattr(args, "matrix_conditions", None) is not None:
        raise GuardRefusal("MATRIX_CONDITIONS_NEEDS_MATRIX", "--matrix-conditions needs --matrix")
    window = getattr(args, "history_window", None)
    state_cap = getattr(args, "state_block_max_bytes", None)
    for flag, value in (("history-window", window), ("state-block-max-bytes", state_cap)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_CAP):
            raise GuardRefusal("INPUT_CAP_OUT_OF_RANGE", f"--{flag}: 0..{MAX_CAP}")
    if not getattr(args, "matrix", False):
        if (window is not None or state_cap is not None) and not getattr(args, "bare_chat", False):
            raise GuardRefusal("INPUT_CAPS_NEED_BARE_CHAT", "--history-window / --state-block-max-bytes need --bare-chat")
        if state_cap is not None and restore is None:
            raise GuardRefusal("STATE_CAP_NEEDS_BARE_RESTORE", "--state-block-max-bytes needs --bare-restore")
    if restore is not None:
        if not getattr(args, "bare_chat", False):
            raise GuardRefusal("BARE_RESTORE_NEEDS_BARE_CHAT", "--bare-restore needs --bare-chat")
        try:
            bare_chat_adapter.parse_restore_list(restore)
        except ValueError as error:
            raise GuardRefusal("BARE_RESTORE_INVALID", str(error)) from error
    elif placement is not None and not getattr(args, "matrix", False):
        raise GuardRefusal("RESTORE_PLACEMENT_NEEDS_BARE_RESTORE", "--restore-placement needs --bare-restore")


def _check_arguments(args: argparse.Namespace) -> None:
    script = getattr(args, "script", "short") or "short"
    if script not in SCRIPTS:
        raise GuardRefusal("SCRIPT_UNKNOWN", f"--script: {', '.join(SCRIPTS)}")
    cap = len(SCRIPTS[script])
    if getattr(args, "turns", None) is None:
        args.turns = cap  # the whole chosen script
    if not 1 <= args.turns <= cap:
        raise GuardRefusal("TURNS_OUT_OF_RANGE", f"1..{cap}")
    if args.repeats is None:  # the matrix defaults to 5 subjects per condition, a plain smoke to 1
        args.repeats = MAX_REPEATS if getattr(args, "matrix", False) else 1
    if not 1 <= args.repeats <= MAX_REPEATS:
        raise GuardRefusal("REPEATS_OUT_OF_RANGE", f"1..{MAX_REPEATS}")
    if args.temperature is not None and not 0.0 <= args.temperature <= MAX_TEMPERATURE:  # NaN fails too
        raise GuardRefusal("TEMPERATURE_OUT_OF_RANGE", f"0.0..{MAX_TEMPERATURE}")
    _check_restore_and_matrix(args)
    if getattr(args, "bare_chat", False):
        if getattr(args, "chat_roles", False):
            raise GuardRefusal("BARE_CHAT_EXCLUDES_CHAT_ROLES", "--bare-chat and --chat-roles are mutually exclusive")
        if not args.record_model_calls:
            raise GuardRefusal("BARE_CHAT_NEEDS_RECORD_MODEL_CALLS", "--bare-chat needs --record-model-calls")
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
        bare_chat=bool(getattr(args, "bare_chat", False)),
        bare_restore=getattr(args, "bare_restore", None),
        restore_placement=getattr(args, "restore_placement", None) or "system",
        history_window=getattr(args, "history_window", None),
        state_block_max_bytes=getattr(args, "state_block_max_bytes", None),
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
        for turn_ordinal, (turn_label, text) in enumerate(script_turns(getattr(args, "script", "short") or "short")[: args.turns], start=1):
            seen = len(read_calls(private_directory)) if args.record_model_calls else 0
            reply = session.chat(text)
            response = str(reply.get("response", ""))
            _keep(private_file, turn_ordinal, turn_label, text, response)
            calls: int | None = None
            sampling: dict[str, Any] | None = None
            assembly: dict[str, Any] | None = None
            roles: dict[str, Any] | None = None
            bare: dict[str, Any] | None = None
            if args.record_model_calls:
                rows = read_calls(private_directory)[seen:]
                calls = len(rows)
                # the call that carried THIS turn's message (the last one whose input holds it)
                carrier = next((row for row in reversed(rows) if text in model_input_text(row)), None)
                if carrier is not None:
                    sampling = carrier.get("sampling")
                    roles = chat_role_adapter.numbers(carrier)
                    bare = bare_chat_adapter.numbers(carrier)
                    assembly = assembly_numbers(
                        carrier,
                        previous_request=None if previous is None else previous[0],
                        previous_response=None if previous is None else previous[1],
                    )
            previous = (text, response)
            turns.append({
                "ordinal": turn_ordinal, "label": turn_label, "seconds": session.turn_seconds[-1],
                "response_chars": len(response), "lyrical_marker_hits": lyrical_marker_hits(response),
                "option_naming": observe_option_naming(turn_label, response),
                "observed_labels": observe_reply(turn_label, response),
                "model_calls": calls, "sampling": sampling, "prompt_assembly": assembly, "chat_roles": roles,
                "bare_chat": bare,
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
        "bare_chat": _bare_record(args, turns),
        "turns": turns,
        "private_file": str(private_file),
        "model_calls_file": str(private_directory / "model_calls.jsonl") if args.record_model_calls else None,
    }
    if getattr(args, "bare_restore", None):  # present only when restores were asked for
        run["restore"] = {
            "sections": list(bare_chat_adapter.parse_restore_list(args.bare_restore)),
            "placement": getattr(args, "restore_placement", None) or "system",
        }
    if getattr(args, "history_window", None) is not None or getattr(args, "state_block_max_bytes", None) is not None:
        run["caps"] = {  # requested; the effective numbers are per turn in bare_chat
            "history_window": getattr(args, "history_window", None),
            "state_block_max_bytes": getattr(args, "state_block_max_bytes", None),
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
            bare = item.get("bare_chat")
            if bare is not None:
                note += (f"  bare chat: {bare['messages_sent_count']} msg sent, history pairs {bare['history_pairs']}, "
                         f"skipped {bare['adapter_skipped']}")
                if "restored_chars" in bare:
                    note += (f", restored {bare['restored_sections']} ({bare['restored_chars']} chars)"
                             + (f", {bare['section_missing']}" if bare["section_missing"] else ""))
                    if "restored_chars_system" in bare:
                        note += (f", placement {bare['restore_placement']} (system {bare['restored_chars_system']} chars, "
                                 f"first user {bare['restored_chars_first_user']} chars)")
            if bare is not None and bare.get("input_tokens") is not None:
                note += f"  input tokens {bare['input_tokens']}"
                if bare.get("peak_allocated_bytes") is not None:
                    note += f", peak allocated {bare['peak_allocated_bytes'] / gpu.GIB:.2f} GiB"
            note += (f"  input: {assembly['messages']} msg, {assembly['prompt_chars']} chars, "
                     f"trace turns {assembly['trace_turns']}, episodic items {assembly['episodic_items']}")
        print(f"turn {item['ordinal']} [{item['label']}]: {item['seconds']:.1f} s, {item['response_chars']} chars, "
              f"lyrical hits {item.get('lyrical_marker_hits', 0)}{note}")
    print(f"raw replies (PRIVATE): {run['private_file']}")


# ------------------------------------------------------------------------------------------ the matrix


@dataclass(frozen=True, slots=True)
class Condition:
    """One matrix condition: a bare chat plus ``restore`` (a comma list or ``None``) at ``placement``."""

    id: str
    description: str
    restore: str | None
    placement: str = "system"
    history_window: int | None = None  # input caps (lab); None = no cap
    state_block_max_bytes: int | None = None


CONDITIONS: tuple[Condition, ...] = (
    Condition("X0", "bare", None),
    Condition("X1", "header", "header"),
    Condition("X2", "emotion", "emotion"),
    Condition("X3", "values", "values"),
    Condition("X4", "wm_evidence", "wm_evidence"),
    Condition("X5", "motivations_goals", "motivations_goals"),
    Condition("X6", "context", "context"),
    Condition("X7", "all (system)", "all", "system"),
    Condition("X8", "all (user, production-like)", "all", "user"),
)
# Opt-in only (``--matrix-conditions``): the #414-compliant placements.  The default matrix stays X0..X8.
EXTRA_CONDITIONS: tuple[Condition, ...] = (
    Condition("X9", "all (first-user)", "all", "first-user"),
    Condition("X10", "all (split-safe)", "all", "split-safe"),
)


_SPEC = re.compile(r"(X[0-9]+)((?::[wb][0-9]+)*)")


def parse_condition(spec: str, known: dict[str, Condition]) -> Condition:
    """``X10`` / ``X10:w4`` / ``X10:b2048`` / ``X10:w4:b2048`` -> the condition with its input caps.

    The base id must be known; each suffix (``w`` = history window, ``b`` = state block bytes, digits only) may occur
    at most once, in either order; ``b`` needs a base that restores something (not X0).  The result's id is the
    canonical spec (``w`` before ``b``).  Anything else is refused.
    """

    match = _SPEC.fullmatch(spec)
    if match is None:
        raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"malformed condition spec: {spec!r}")
    base = known.get(match.group(1))
    if base is None:
        raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"unknown condition id: {match.group(1)}")
    caps: dict[str, int] = {}
    for part in [piece for piece in match.group(2).split(":") if piece]:
        if part[0] in caps:
            raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"{spec}: duplicate :{part[0]} suffix")
        caps[part[0]] = int(part[1:])
    if any(value > MAX_CAP for value in caps.values()):
        raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"{spec}: cap above {MAX_CAP}")
    if "b" in caps and base.restore is None:
        raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"{spec}: :b needs a condition that restores sections")
    if not caps:
        return base
    canonical = base.id + (f":w{caps['w']}" if "w" in caps else "") + (f":b{caps['b']}" if "b" in caps else "")
    notes = ([f"window {caps['w']}"] if "w" in caps else []) + ([f"state <= {caps['b']} B"] if "b" in caps else [])
    return Condition(
        canonical, f"{base.description} [{', '.join(notes)}]", base.restore, base.placement,
        caps.get("w"), caps.get("b"),
    )


def resolve_conditions(text: str | None) -> tuple[Condition, ...]:
    """``--matrix-conditions`` (a comma list of ids, run in the given order) -> the conditions.

    ``None`` is the default list X0..X8.  A condition id may carry the input-cap suffixes ``:w<N>`` / ``:b<bytes>``
    (see ``parse_condition``).  An unknown id, a malformed spec, an empty item or a duplicate is refused.
    """

    if text is None:
        return CONDITIONS
    known = {condition.id: condition for condition in (*CONDITIONS, *EXTRA_CONDITIONS)}
    specs = [part.strip() for part in text.split(",")]
    if any(not name for name in specs):
        raise GuardRefusal("MATRIX_CONDITIONS_INVALID", "empty or duplicate condition id")
    conditions = tuple(parse_condition(spec, known) for spec in specs)
    if len({condition.id for condition in conditions}) != len(conditions):
        raise GuardRefusal("MATRIX_CONDITIONS_INVALID", "empty or duplicate condition id")
    return conditions


# (column title, turn label, observed key): the counts the table prints
KEY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("E one", "E", "names_exactly_one"),
    ("reph tokyo", "rephrase", "mentions_tokyo"),
    ("pop number", "population", "answers_with_number"),
    ("recall both", "recall", "has_both_phrases"),
)


def check_conditions(conditions: tuple[Condition, ...], repeats: int) -> None:
    """The fixed list must be well formed and the total number of subjects within the cap."""

    ids = [condition.id for condition in conditions]
    if not conditions or len(set(ids)) != len(ids):
        raise GuardRefusal("MATRIX_CONDITIONS_INVALID", "empty or duplicate condition ids")
    for condition in conditions:
        if condition.placement not in bare_chat_adapter.PLACEMENTS:
            raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"{condition.id}: placement")
        for cap in (condition.history_window, condition.state_block_max_bytes):
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or not 0 <= cap <= MAX_CAP):
                raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"{condition.id}: input cap")
        if condition.state_block_max_bytes is not None and condition.restore is None:
            raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"{condition.id}: state cap without restore")
        if condition.restore is not None:
            try:
                bare_chat_adapter.parse_restore_list(condition.restore)
            except ValueError as error:
                raise GuardRefusal("MATRIX_CONDITIONS_INVALID", f"{condition.id}: {error}") from error
    if len(conditions) * repeats > MATRIX_MAX_SUBJECTS:
        raise GuardRefusal("MATRIX_TOO_MANY_SUBJECTS", f"{len(conditions)} x {repeats} > {MATRIX_MAX_SUBJECTS}")


def _observed_count(aggregated: dict[str, Any], label: str, key: str) -> dict[str, int] | None:
    observed = (aggregated.get("per_label", {}).get(label) or {}).get("observed") or {}
    count = observed.get(key)
    return count if isinstance(count, dict) and "true" in count else None


def _condition_summary(condition: Condition, runs: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    completed = [run for run in runs if run["ended"] == "completed"]
    aggregated = aggregate(completed)
    free = [int(run["gpu"]["min_free"]) for run in runs if run.get("gpu", {}).get("min_free") is not None]
    bare_rows = [turn["bare_chat"] for run in completed for turn in run["turns"] if turn.get("bare_chat")]

    def mean(name: str) -> float | None:
        values = [row[name] for row in bare_rows if isinstance(row.get(name), (int, float))]
        return round(sum(values) / len(values), 1) if values else None

    def peak(name: str) -> int | None:
        values = [row[name] for row in bare_rows if isinstance(row.get(name), int)]
        return max(values) if values else None

    gpu_peaks = [int(run["gpu"]["peak_allocated"]) for run in runs if run.get("gpu", {}).get("peak_allocated") is not None]
    turn_peaks = peak("peak_allocated_bytes")
    peak_allocated = max([*gpu_peaks, *([] if turn_peaks is None else [turn_peaks])], default=None)
    split = {}
    if condition.restore and condition.placement in bare_chat_adapter.FIRST_USER_PLACEMENTS:
        # numbers per destination (system vs first user turn); present only for these placements
        split = {"mean_restored_chars_system": mean("restored_chars_system"),
                 "mean_restored_chars_first_user": mean("restored_chars_first_user")}
    return {
        **split,
        "id": condition.id, "description": condition.description, "restore": condition.restore,
        "placement": condition.placement if condition.restore else None,
        "subjects_planned": planned, "subjects_started": len(runs), "completed_runs": len(completed),
        "ended": [run["ended"] for run in runs],
        "min_free_bytes": min(free) if free else None,
        "history_window": condition.history_window, "state_block_max_bytes": condition.state_block_max_bytes,
        "max_input_tokens": peak("input_tokens"), "peak_allocated_bytes": peak_allocated,
        "max_reserved_bytes": peak("reserved_bytes"),
        "mean_input_tokens": mean("input_tokens"), "mean_exchanges_dropped": mean("exchanges_dropped"),
        "mean_sections_dropped": mean("sections_dropped"), "mean_state_bytes_effective": mean("state_bytes_effective"),
        "mean_user_chars_after": mean("user_chars_after"), "mean_input_chars": mean("input_chars"),
        "mean_restored_chars": mean("restored_chars"),
        "section_missing_turns": sum(1 for row in bare_rows if row.get("section_missing")),
        "persona": aggregated["persona"],
        "key_counts": {title: _observed_count(aggregated, label, key) for title, label, key in KEY_COLUMNS},
        "aggregate": aggregated,
        "runs": runs,
    }


def matrix_table(summaries: list[dict[str, Any]]) -> list[str]:
    """The compact table (rows = conditions): counts as ``true/of``, ``-`` where nothing was observed."""

    header = ["cond", "what", "done", *[title for title, _l, _k in KEY_COLUMNS], "lyric hits", "mean chars", "min free GiB", "peak alloc GiB", "max input tokens", "input chars"]
    rows = [header]
    for item in summaries:
        counts = [
            "-" if item["key_counts"][title] is None else f"{item['key_counts'][title]['true']}/{item['key_counts'][title]['of']}"
            for title, _l, _k in KEY_COLUMNS
        ]
        free = "-" if item["min_free_bytes"] is None else f"{item['min_free_bytes'] / gpu.GIB:.1f}"
        allocated = "-" if item.get("peak_allocated_bytes") is None else f"{item['peak_allocated_bytes'] / gpu.GIB:.1f}"
        tokens = "-" if item.get("max_input_tokens") is None else str(item["max_input_tokens"])
        persona = item.get("persona") or {}
        lyric = "-" if persona.get("mean_lyrical_marker_hits") is None else f"{persona['mean_lyrical_marker_hits']:.2f}"
        reply = "-" if persona.get("mean_response_chars") is None else f"{persona['mean_response_chars']:.0f}"
        chars = "-" if item["mean_input_chars"] is None else f"{item['mean_input_chars']:.0f}"
        rows.append([item["id"], item["description"], f"{item['completed_runs']}/{item['subjects_planned']}", *counts, lyric, reply, free, allocated, tokens, chars])
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def _write_matrix_results(base: Path, launch_id: str, payload: dict[str, Any]) -> Path:
    """Rewrite the matrix results atomically (0600); called after every subject so an interrupt keeps the rest."""

    path = Path(base) / f"results-smoke-matrix-{launch_id}.json"
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=1, default=str)
    os.replace(temporary, path)
    return path


def _matrix_summary_lines(args: argparse.Namespace, model: ModelOverride | None, conditions: tuple[Condition, ...]) -> list[str]:
    return [
        *_summary_lines(args, model),
        f"MATRIX: {len(conditions)} conditions x {args.repeats} subjects = {len(conditions) * args.repeats} fresh subjects "
        f"(cap {MATRIX_MAX_SUBJECTS}), sequential, bare chat with restored sections: "
        + ", ".join(f"{c.id}={c.description}" for c in conditions),
        "restored sections go only to the model and the private model_calls.jsonl; results hold numbers and labels only",
    ]


def _matrix(
    args: argparse.Namespace, *, attest: Attest, gpu_runner: Callable[[], str] | None,
    conditions: tuple[Condition, ...] | None, gate: Callable[[], None] | None,
) -> int:
    """The condition matrix under ONE Human attestation.  ``conditions`` / ``gate`` are test seams only."""

    chosen = resolve_conditions(getattr(args, "matrix_conditions", None)) if conditions is None else conditions
    check_conditions(chosen, args.repeats)
    args = argparse.Namespace(**vars(args))
    args.bare_chat = True  # the matrix IS the bare chat with restores
    if args.temperature is None:
        args.temperature = 0.0  # greedy unless the Human asked for a temperature
    base = Path(args.base)
    check_not_production(base)  # BEFORE anything is created
    model = _model(args)
    launch: HumanLaunch | None = None
    reading: gpu.GpuReading | None = None
    revision = ""
    if model is not None:
        revision = check_pinned_revision(model)
        assert model.revision is not None

        def before(_terminal: Terminal) -> list[str]:
            return _matrix_summary_lines(args, model, chosen)

        launch = attest(
            STAGE, model_digest(model), human_flag=bool(args.human_launch), revision8=model.revision[:8],
            before_confirm=before,
        )
        reading = gpu.check_free(runner=gpu_runner)
    launch_id = uuid.uuid4().hex[:10]
    fixture = model is None
    summaries: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None
    code = 0
    interrupted = False

    def payload(current: dict[str, Any] | None) -> dict[str, Any]:
        shown = [*summaries, *([] if current is None else [current])]
        return {
            "kind": "observations", "verdict": None, "fixture_only": fixture, "launch": launch_id, "matrix": True,
            "requested": {"repeats": args.repeats, "turns": args.turns, "script": getattr(args, "script", "short") or "short",
                          "conditions": [c.id for c in chosen],
                          "subjects_planned": len(chosen) * args.repeats, "record_model_calls": bool(args.record_model_calls)},
            "generation": {"temperature_requested": args.temperature, "temperature_zero_means_greedy": True,
                           "do_sample_override": args.temperature != 0.0},
            "conditions": shown,
            "completed_conditions": [item["id"] for item in shown if item["completed_runs"] == item["subjects_planned"]],
            "stopped": stopped, "interrupted": interrupted,
        }

    started = 0
    try:
        for condition in chosen:
            runs: list[dict[str, Any]] = []
            subject_args = argparse.Namespace(**vars(args))
            subject_args.bare_restore = condition.restore
            subject_args.restore_placement = condition.placement if condition.restore else None
            subject_args.history_window = condition.history_window
            subject_args.state_block_max_bytes = condition.state_block_max_bytes
            print(f"--- condition {condition.id}: {condition.description} ---")
            for ordinal in range(1, args.repeats + 1):
                started += 1
                if started > 1:  # the first subject follows the launch check
                    try:
                        if gate is not None:
                            gate()
                        elif model is not None:
                            reading = gpu.check_free(runner=gpu_runner)  # before EVERY subject
                    except GuardRefusal as refusal:
                        stopped = {"condition": condition.id, "before_run": ordinal, "code": refusal.code, "detail": refusal.detail}
                        code = EXIT_REFUSED
                        print(f"stopped before {condition.id} run {ordinal}: {refusal.code} {refusal.detail}".rstrip())
                        break
                run, run_code = _one_run(
                    subject_args, base, launch_id, ordinal, model=model, launch=launch, reading=reading, revision=revision,
                )
                runs.append(run)
                _print_run(run, reading)
                _write_matrix_results(base, launch_id, payload(_condition_summary(condition, runs, args.repeats)))
                if run_code:
                    code = run_code
                    stopped = {"condition": condition.id, "before_run": ordinal + 1, "code": run["ended"],
                               "detail": "the previous run did not complete"}
                    break
            if runs:
                summaries.append(_condition_summary(condition, runs, args.repeats))
            if stopped is not None:
                break
    except KeyboardInterrupt:
        interrupted, code = True, EXIT_FAILED
    results = _write_matrix_results(base, launch_id, payload(None))
    print(f"matrix: script {getattr(args, 'script', 'short') or 'short'}, {args.turns} turns, temperature {args.temperature}"
          "; lyric hits / mean chars = mean per reply over all turns (observed, not a verdict)")
    print("\n".join(matrix_table(summaries)))
    done = [item["id"] for item in summaries if item["completed_runs"] == item["subjects_planned"]]
    print(f"completed conditions: {', '.join(done) or 'none'}" + (f"; stopped: {stopped['code']}" if stopped else "")
          + ("; interrupted" if interrupted else ""))
    print(f"results (numbers and labels only): {results}")
    print(artifacts.RETENTION_NOTE.strip())
    print("These are observations, not a verdict on the model.")
    return code


def _smoke(
    args: argparse.Namespace, *, attest: Attest, gpu_runner: Callable[[], str] | None,
    conditions: tuple[Condition, ...] | None = None, gate: Callable[[], None] | None = None,
) -> int:
    """The smoke.  PRIVATE seam version (``attest`` / ``gpu_runner``): only tests pass anything else;
    ``run`` below is hard-wired to the real terminal and the real GPU probe."""

    _check_arguments(args)
    if getattr(args, "matrix", False):
        return _matrix(args, attest=attest, gpu_runner=gpu_runner, conditions=conditions, gate=gate)
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
        "bare_chat": {
            "requested": bool(getattr(args, "bare_chat", False)),
            "applied": bool(completed) and all(run.get("bare_chat", {}).get("applied") is True for run in completed),
            "turns_bare": sum(run.get("bare_chat", {}).get("turns_bare", 0) for run in runs),
            "turns_passed_through": sum(run.get("bare_chat", {}).get("turns_passed_through", 0) for run in runs),
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
    top.add_argument("--turns", type=int, default=None,
                     help=f"number of chat turns: 1..{MAX_TURNS} for --script short, 1..{LONG_MAX_TURNS} for long (default: all)")
    top.add_argument("--script", choices=list(SCRIPTS), default="short",
                     help=f"the fixed turn script: short ({MAX_TURNS} turns, default) or long ({LONG_MAX_TURNS} turns: probes at turns "
                          "13-16, 12+ turns after the first-turn state block; E at turn 2); with fewer --turns the end probes are cut")
    top.add_argument("--temperature", type=float,
                     help=f"override generation.temperature, 0.0..{MAX_TEMPERATURE} (0 = greedy); default: the config value")
    top.add_argument("--repeats", type=int, default=None,
                     help=f"sequential fresh subjects under one attestation, 1..{MAX_REPEATS} (default 1; with --matrix: 5 per condition)")
    top.add_argument("--record-model-calls", action="store_true",
                     help="lab diagnostics: record every model call privately and report prompt-assembly numbers")
    top.add_argument("--conversation-trace", action="store_true",
                     help="lab diagnostics: enable conversation_trace in the throwaway subject's config (default off)")
    top.add_argument("--bare-chat", action="store_true",
                     help="lab diagnostics control: the model sees only a neutral system text, its own history and the current message (needs --record-model-calls; not with --chat-roles)")
    top.add_argument("--bare-restore", metavar="LIST",
                     help="lab diagnostics: restore these sections of the real prompt into the bare chat, verbatim "
                          f"(comma list of {', '.join(bare_chat_adapter.RESTORE_NAMES)}, or all; needs --bare-chat)")
    top.add_argument("--restore-placement", choices=list(bare_chat_adapter.PLACEMENTS),
                     help="where restored sections go: system (appended to the bare system text; default), user (prepended inside the "
                          "final user message), first-user (a labelled data block in the FIRST user turn of the conversation, "
                          "re-attached every call; the final user message is the user's text only) or split-safe (code-generated "
                          "sections header/emotion/context in the system text, values/wm_evidence/motivations_goals in the first user turn; #414)")
    top.add_argument("--history-window", type=int, metavar="N",
                     help="lab input cap (needs --bare-chat): send only the most recent N exchanges of the adapter's own history "
                          "(N >= 0; default: all); the user text and assistant text are never truncated")
    top.add_argument("--state-block-max-bytes", type=int, metavar="B",
                     help="lab input cap (needs --bare-restore): drop whole restored sections from the END until the state block "
                          "is at most B UTF-8 bytes (default: no cap)")
    top.add_argument("--matrix-conditions", metavar="IDS",
                     help="with --matrix: a comma list of condition ids to run, in this order (default X0..X8; "
                          "X9 = all first-user, X10 = all split-safe are available only here), e.g. X0,X7,X9,X10; an id may "
                          "carry input-cap suffixes :w<N> (history window) and/or :b<bytes> (state block cap), "
                          "e.g. X10:w4:b2048")
    top.add_argument("--matrix", action="store_true",
                     help="lab diagnostics: run the fixed condition matrix X0..X8 (bare + restores) in ONE launch, "
                          f"--repeats subjects each; total subjects <= {MATRIX_MAX_SUBJECTS}; greedy unless --temperature; "
                          "needs --record-model-calls; estimated 2.5-4.5 h for 45 subjects (assumption, not measured)")
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


__all__ = ["LONG_MAX_TURNS", "LONG_TURNS", "LYRICAL_MARKERS", "OPTION_PHRASES", "SCRIPTS", "TURNS", "lyrical_marker_hits", "script_turns", "aggregate", "main", "observe_option_naming", "observe_reply", "run"]
