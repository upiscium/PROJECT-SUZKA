"""Isolated R15 P1 candidate experiment. No authoritative Suzka state is changed.

S1 is captured and fsynced before the transport receives E2. S1/S2 are
model-supplied self-interpretation *candidates*, NOT NarrativeSelf admissions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, NoReturn

from experiments.r15_p0.pilot import (
    atomic_private_create,
    canonical_bytes,
    http_chat,
    read_sealed,
)


AUTHORITY = "P1_TEST_ONLY_CANDIDATES_NOT_R15"
MAX_RESPONSE_BYTES = 4096
MAX_PROMPT_BYTES = 8192
MAX_FILE_BYTES = 65536
MAX_RATIONALE_BYTES = 700
_STEPS = ("setup", "E", "S1", "E2", "S2")
_CHOICES = frozenset(("A", "B", "UNKNOWN"))
_ASSESSMENTS = frozenset(("LIKELY_ALIGNED", "MAY_HAVE_MISREAD", "UNKNOWN"))
_UNCERTAINTIES = frozenset(("LOW", "MEDIUM", "HIGH"))


class TrialRefused(ValueError):
    """Coarse failure; does not include private transcript, paths or IDs."""


def _refuse(reason: str) -> NoReturn:
    raise TrialRefused(reason)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            _refuse("duplicate JSON key")
        out[key] = value
    return out


def _no_constants(_: str) -> NoReturn:
    _refuse("invalid JSON constant")


def _strict_json(raw: bytes) -> Any:
    if len(raw) > MAX_FILE_BYTES:
        _refuse("record exceeds bound")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constants,
        )
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise TrialRefused("malformed JSON record") from exc


def _read(path: Path) -> tuple[Any, bytes]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise TrialRefused("record unavailable") from exc
    return _strict_json(raw), raw


def _bounded(value: object, maximum: int, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        _refuse("invalid text type")
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise TrialRefused("invalid UTF-8 string") from exc
    if length > maximum or (nonempty and length == 0) or "\x00" in value:
        _refuse("text exceeds permitted bound")
    return value


@dataclass(frozen=True, slots=True)
class Candidate:
    phase: str
    assumed_referent: str
    assessment: str
    uncertainty: str
    evidence_steps: tuple[str, ...]
    rationale: str

    def semantic_value(self) -> tuple[str, str, str, str]:
        return (
            self.assumed_referent,
            self.assessment,
            self.uncertainty,
            self.rationale,
        )


def parse_candidate(response: str, *, phase: str, has_reveal: bool) -> Candidate:
    """Treat a model self-report as untrusted, closed, finite *candidate* data."""
    _bounded(response, MAX_RESPONSE_BYTES)
    data = _strict_json(response.encode("utf-8"))
    if type(data) is not dict or set(data) != {
        "schema",
        "phase",
        "assumed_referent",
        "assessment",
        "uncertainty",
        "evidence_steps",
        "rationale",
    }:
        _refuse("noncanonical candidate fields")
    expected_sources = (
        ("E",) if phase == "S1" else ("E", "S1", "E2") if has_reveal else ("E", "S1")
    )
    if (
        type(data["schema"]) is not int
        or data["schema"] != 1
        or phase not in ("S1", "S2")
        or type(data["phase"]) is not str
        or data["phase"] != phase
        or type(data["assumed_referent"]) is not str
        or data["assumed_referent"] not in _CHOICES
        or type(data["assessment"]) is not str
        or data["assessment"] not in _ASSESSMENTS
        or type(data["uncertainty"]) is not str
        or data["uncertainty"] not in _UNCERTAINTIES
        or type(data["evidence_steps"]) is not list
        or tuple(data["evidence_steps"]) != expected_sources
    ):
        _refuse("candidate stage or evidence mismatch")
    rationale = _bounded(data["rationale"], MAX_RATIONALE_BYTES)
    return Candidate(
        phase,
        data["assumed_referent"],
        data["assessment"],
        data["uncertainty"],
        expected_sources,
        rationale,
    )


_S1_REQUEST = (
    "これは隔離実験であり、正式な記憶・信頼・能力判定ではありません。"
    "現時点で見た会話と自分の直前の回答だけから、相手の「前の案」の意味を"
    "自分がどう暫定的に解釈したか、誤解した可能性も含めて候補を記述してください。"
    "今後の補足はまだありません。断定できなければUNKNOWNとしてください。"
    "JSONオブジェクトだけを返してください。キーは"
    "schema(数値1),phase(文字列S1),assumed_referent(A/B/UNKNOWN),"
    "assessment(LIKELY_ALIGNED/MAY_HAVE_MISREAD/UNKNOWN),"
    'uncertainty(LOW/MEDIUM/HIGH),evidence_steps(["E"]),'
    "rationale(短い日本語)です。"
)


def s1_request() -> str:
    """Independent of case ID, hidden manifest target and future E2 bytes."""
    return _S1_REQUEST


def s2_request(s1_response: str, *, has_reveal: bool) -> str:
    _bounded(s1_response, MAX_RESPONSE_BYTES)
    evidence = '["E","S1","E2"]' if has_reveal else '["E","S1"]'
    context = (
        "後からユーザーの補足E2がありました。ただしユーザーの発言は"
        "発言の証拠であって、第三者の客観的な意図の証明ではありません。"
        if has_reveal
        else "後続の補足E2は存在しません。未提示の参照先を確定しないでください。"
    )
    result = (
        "これは隔離実験の自己解釈候補です。正式なNarrativeSelfや信頼を"
        "更新したと主張しないでください。元の経験Eと下記のS1を保持したまま、"
        "現時点の解釈候補S2をJSONのみで返してください。"
        + context
        + " schema(数値1),phase(文字列S2),assumed_referent(A/B/UNKNOWN),"
        "assessment(LIKELY_ALIGNED/MAY_HAVE_MISREAD/UNKNOWN),"
        "uncertainty(LOW/MEDIUM/HIGH),evidence_steps("
        + evidence
        + "),rationale(短い日本語)の7キーだけです。"
        "S1を上書きせず、変わらない場合は無理に変化を作らないでください。"
        "先行するS1候補:\n" + s1_response
    )
    _bounded(result, MAX_PROMPT_BYTES)
    return result


def _paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.name + ".started.json"),
        output.with_name(output.name + ".s1.json"),
    )


def _create(output: Path, record: dict[str, Any]) -> None:
    """P0's create-only/file-fsync followed by explicit parent-dir fsync.

    A filesystem/power-cut certification is NOT supplied by this operation.
    If directory fsync fails, retain the stage as ambiguous; never retry.
    """
    try:
        atomic_private_create(output, record)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise TrialRefused("attempted publication uncertain") from exc


def _row(
    step: str,
    request: str,
    transport: Callable[[str], dict[str, str]],
    index: int,
    seen: set[str],
) -> dict[str, Any]:
    _bounded(request, MAX_PROMPT_BYTES)
    value = transport(request)
    if type(value) is not dict or set(value) != {"episode_id", "response"}:
        _refuse("transport shape unavailable")
    identifier = _bounded(value["episode_id"], 256)
    response = _bounded(value["response"], MAX_RESPONSE_BYTES)
    if identifier in seen:
        _refuse("duplicate episode identity")
    seen.add(identifier)
    return {
        "step": step,
        "ordinal": index,
        "request": request,
        "episode_id": identifier,
        "response": response,
    }


def _checkpoint(
    case_id: str, manifest_seal: str, turns: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(turns) != 3:
        _refuse("S1 checkpoint must precede E2")
    return {
        "schema": 1,
        "authority": AUTHORITY,
        "case_id": case_id,
        "manifest_seal": manifest_seal,
        "turns": turns,
    }


def run_trial(
    manifest_path: Path,
    output_path: Path,
    transport: Callable[[str], dict[str, str]],
) -> dict[str, Any]:
    """Observe real transport-supplied S1, durably store it, then disclose E2."""
    if not callable(transport):
        _refuse("transport unavailable")
    case, manifest = read_sealed(manifest_path)
    started_path, s1_path = _paths(output_path)
    if any(
        path.exists() or path.is_symlink()
        for path in (started_path, s1_path, output_path)
    ):
        _refuse("trial already started; never retry")
    _create(
        started_path,
        {
            "schema": 1,
            "authority": AUTHORITY,
            "case_id": case.case_id,
            "manifest_seal": manifest["seal"],
            "state": "STARTED_NOT_R15_ADMISSION",
        },
    )
    prompts = case.prompts()
    seen: set[str] = set()
    turns = [
        _row("setup", prompts["setup"], transport, 0, seen),
        _row("E", prompts["E"], transport, 1, seen),
    ]
    turns.append(_row("S1", s1_request(), transport, 2, seen))
    parse_candidate(turns[-1]["response"], phase="S1", has_reveal=False)
    stage = _checkpoint(case.case_id, manifest["seal"], turns.copy())
    _create(s1_path, stage)
    saved, raw = _read(s1_path)
    if saved != stage or raw != canonical_bytes(stage) + b"\n":
        _refuse("S1 durable checkpoint mismatch")
    stage_digest = hashlib.sha256(raw).hexdigest()
    if case.control != "no_reveal":
        turns.append(_row("E2", prompts["reveal"], transport, 3, seen))
    turns.append(
        _row(
            "S2",
            s2_request(
                stage["turns"][2]["response"], has_reveal=case.control != "no_reveal"
            ),
            transport,
            len(turns),
            seen,
        )
    )
    parse_candidate(
        turns[-1]["response"], phase="S2", has_reveal=case.control != "no_reveal"
    )
    result = {
        "schema": 1,
        "authority": AUTHORITY,
        "case_id": case.case_id,
        "manifest_seal": manifest["seal"],
        "s1_checkpoint_sha256": stage_digest,
        "turns": turns,
    }
    _create(output_path, result)
    audit_trial(manifest_path, output_path)
    return result


def _check_row(
    item: Any,
    step: str,
    ordinal: int,
    expected_request: str,
    seen: set[str],
) -> None:
    if (
        type(item) is not dict
        or set(item) != {"step", "ordinal", "request", "episode_id", "response"}
        or item["step"] != step
        or type(item["ordinal"]) is not int
        or item["ordinal"] != ordinal
        or item["request"] != expected_request
    ):
        _refuse("turn chronology or request changed")
    identifier = _bounded(item["episode_id"], 256)
    _bounded(item["response"], MAX_RESPONSE_BYTES)
    if identifier in seen:
        _refuse("duplicate episode identity")
    seen.add(identifier)


def audit_trial(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Verify stage chronology and immutable copies, not first-person truth."""
    case, manifest = read_sealed(manifest_path)
    started_path, s1_path = _paths(output_path)
    started, _ = _read(started_path)
    if type(started) is not dict or started != {
        "schema": 1,
        "authority": AUTHORITY,
        "case_id": case.case_id,
        "manifest_seal": manifest["seal"],
        "state": "STARTED_NOT_R15_ADMISSION",
    }:
        _refuse("started attempt mismatched")
    stage, stage_raw = _read(s1_path)
    trial, trial_raw = _read(output_path)
    if (
        len(trial_raw) > MAX_FILE_BYTES
        or type(trial) is not dict
        or set(trial)
        != {
            "schema",
            "authority",
            "case_id",
            "manifest_seal",
            "s1_checkpoint_sha256",
            "turns",
        }
    ):
        _refuse("invalid final record fields")
    expected_stages = (
        ("setup", "E", "S1", "S2") if case.control == "no_reveal" else _STEPS
    )
    turns = trial["turns"]
    if type(turns) is not list or any(type(row) is not dict for row in turns):
        _refuse("invalid final stage records")
    if (
        type(trial["schema"]) is not int
        or trial["schema"] != 1
        or trial["authority"] != AUTHORITY
        or trial["case_id"] != case.case_id
        or trial["manifest_seal"] != manifest["seal"]
        or type(turns) is not list
        or len(turns) != len(expected_stages)
    ):
        _refuse("final manifest or stage mismatch")
    expect_stage = _checkpoint(case.case_id, manifest["seal"], turns[:3])
    if stage != expect_stage or stage_raw != canonical_bytes(expect_stage) + b"\n":
        _refuse("S1 has changed or checkpoint missing")
    digest = hashlib.sha256(stage_raw).hexdigest()
    if trial["s1_checkpoint_sha256"] != digest:
        _refuse("S1 digest mismatch")
    if trial_raw != canonical_bytes(trial) + b"\n":
        _refuse("final record is not canonical")
    seen: set[str] = set()
    prompts = case.prompts()
    s1 = turns[2]
    for index, step in enumerate(expected_stages):
        expected = (
            prompts[step]
            if step in ("setup", "E")
            else s1_request()
            if step == "S1"
            else prompts["reveal"]
            if step == "E2"
            else s2_request(s1["response"], has_reveal=case.control != "no_reveal")
        )
        _check_row(turns[index], step, index, expected, seen)
    c1 = parse_candidate(s1["response"], phase="S1", has_reveal=False)
    c2 = parse_candidate(
        turns[-1]["response"], phase="S2", has_reveal=case.control != "no_reveal"
    )
    return {
        "status": AUTHORITY,
        "s1_before_e2_checkpoint": True,
        "has_e2": case.control != "no_reveal",
        "pre_e2_blind": case.control != "explicit",
        "candidate_changed": c1.semantic_value() != c2.semantic_value(),
        "episode_count": len(turns),
        "production_r15_admission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic P1 candidate chronology; NOT R15 admission"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("capture", type=Path)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--isolated-instance", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("capture", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if not args.isolated_instance:
            parser.error(
                "an independently provisioned fresh isolated Suzka instance is required"
            )
        run_trial(
            args.manifest, args.capture, lambda msg: http_chat(args.base_url, msg)
        )
    print(json.dumps(audit_trial(args.manifest, args.capture), ensure_ascii=False))


if __name__ == "__main__":
    main()
