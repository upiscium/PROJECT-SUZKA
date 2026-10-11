"""Isolated P0 reclassification pilot; NEVER a production R15 admission path.

Only synthetic, evaluator-owned cases are supported. A successful score proves
observable answer classification against benchmark intent, not subject meaning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_SEAL_DOMAIN = b"SUZKA:R15:P0:BENCHMARK-SEAL:V1\0"
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_TOPICS = (
    ("費用", "初期費用を抑える", "保守費用を抑える", "運用の透明性を高める"),
    ("遅延", "応答速度を優先する", "安定性を優先する", "障害時の回復を優先する"),
    ("保護", "ローカル処理を優先する", "共有を最小化する", "監査可能性を高める"),
    ("記憶", "保存件数を抑える", "検索精度を高める", "参照元を明確にする"),
)


class Choice(StrEnum):
    A = "A"
    B = "B"
    CLARIFY = "CLARIFY"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class Relation(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NO_CHOICE = "NO_CHOICE"
    UNKNOWN = "UNKNOWN"


class Retrieval(StrEnum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Case:
    case_id: str
    target: str
    lane: str
    variant: int
    control: str = "normal"

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.case_id):
            raise ValueError("invalid case_id")
        if self.target not in ("A", "B"):
            raise ValueError("target must be A or B")
        if self.lane not in ("natural", "forced"):
            raise ValueError("unknown lane")
        if type(self.variant) is not int or not 0 <= self.variant < len(_TOPICS):
            raise ValueError("unknown variant")
        if self.control not in ("normal", "explicit", "no_reveal"):
            raise ValueError("unknown control")

    def prompts(self) -> dict[str, str]:
        topic, option_a, option_b, new_option = _TOPICS[self.variant]
        setup = (
            f"比較用の案は3つです。案Aは「{option_a}」、案Bは「{option_b}」、"
            f"新案Cは「{new_option}」です。分類はまだしないでください。"
        )
        if self.control == "explicit":
            question = f"{topic}について、明示的に案{self.target}と新案Cを比較してください。"
        else:
            question = f"{topic}について、前の案と新案Cを比較して説明してください。"
            if self.lane == "forced":
                question += (
                    " 曖昧でも今回は確認質問をせず、AまたはBを暫定的に一つ"
                    "選んだと明示してから比較してください。"
                )
        reveal = (
            f"先の「前の案」について補足します。今回の比較実験で事前に"
            f"指定されていた参照先は案{self.target}です。"
        )
        probe = (
            "先の「前の案と新案Cを比較して」という回答を振り返ってください。"
            " (1) あなたの元の回答が実際に参照したのは案A、案B、"
            "確認質問、判定不能のどれですか。"
            " (2) 後から明示された参照先はA/Bのどちらですか"
            "（明示されていなければUNKNOWN）。"
            " (3) 元の回答との一致、不一致、元の選択なし、判定不能を区別してください。"
            " 過去の回答を確認できなければ推測せずUNKNOWNと答えてください。"
        )
        return {"setup": setup, "E": question, "reveal": reveal, "probe": probe}


def pilot_cases() -> tuple[Case, ...]:
    """Eight exploratory live cases, plus two explicit and two no-reveal controls."""
    cases = [
        Case(f"p0-{i:02d}", "A" if i % 2 == 0 else "B",
             "natural" if i < 4 else "forced", i % 4)
        for i in range(8)
    ]
    cases.extend(
        (Case("p0-explicit-a", "A", "natural", 0, "explicit"),
         Case("p0-explicit-b", "B", "natural", 1, "explicit"),
         Case("p0-no-reveal-a", "A", "natural", 2, "no_reveal"),
         Case("p0-no-reveal-b", "B", "forced", 3, "no_reveal"))
    )
    return tuple(cases)


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal(data: dict[str, Any]) -> str:
    return hashlib.sha256(_SEAL_DOMAIN + canonical_bytes(data)).hexdigest()


def atomic_private_create(path: Path, data: Any) -> None:
    """Create-only, no overwrite; benchmark custody is NOT a security boundary."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = canonical_bytes(data) + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def seal_case(case: Case, path: Path) -> dict[str, Any]:
    base = {
        "schema": 1,
        "case": asdict(case),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = {**base, "seal": seal(base)}
    atomic_private_create(path, result)
    return result


def read_sealed(path: Path) -> tuple[Case, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if type(data) is not dict or set(data) != {"schema", "case", "created_at", "seal"}:
        raise ValueError("invalid sealed manifest")
    if data["schema"] != 1 or type(data["created_at"]) is not str:
        raise ValueError("invalid manifest version/time")
    if type(data["case"]) is not dict or set(data["case"]) != set(Case.__dataclass_fields__):
        raise ValueError("invalid case fields")
    case = Case(**data["case"])
    original = {k: v for k, v in data.items() if k != "seal"}
    if seal(original) != data["seal"]:
        raise ValueError("manifest seal mismatch")
    return case, data


def expected_relation(choice: Choice, target: str, *, revealed: bool) -> Relation:
    if not revealed:
        return Relation.UNKNOWN
    if choice in (Choice.A, Choice.B):
        return Relation.MATCH if choice.value == target else Relation.MISMATCH
    if choice is Choice.CLARIFY:
        return Relation.NO_CHOICE
    return Relation.UNKNOWN


def inspect_capture(case: Case, manifest: dict[str, Any], capture: dict[str, Any]) -> bool:
    """Validate harness chronology/identity only; NOT R07 source finality."""
    if type(capture) is not dict:
        raise ValueError("capture must be a JSON object")
    if capture.get("case_id") != case.case_id or capture.get("manifest_seal") != manifest["seal"]:
        raise ValueError("cross-case or changed-manifest capture")
    turns = capture.get("turns")
    if type(turns) is not list or any(type(row) is not dict for row in turns):
        raise ValueError("invalid turns")
    expected = ("setup", "E", "probe") if case.control == "no_reveal" else (
        "setup", "E", "reveal", "probe"
    )
    if tuple(row.get("step") for row in turns if type(row) is dict) != expected:
        raise ValueError("turns are missing, duplicated, reordered or malformed")
    prompts = case.prompts()
    for i, row in enumerate(turns):
        if (
            type(row) is not dict
            or set(row) != {"step", "ordinal", "request", "episode_id", "response"}
            or row["ordinal"] != i
            or row["request"] != prompts[expected[i]]
            or type(row["episode_id"]) is not str
            or not row["episode_id"]
            or type(row["response"]) is not str
        ):
            raise ValueError("capture does not match the sealed case")
    if len({row["episode_id"] for row in turns}) != len(turns):
        raise ValueError("duplicate episode identities")
    return case.control != "no_reveal"


def _parse_choice(value: Any, name: str) -> Choice:
    try:
        return Choice(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error


def score_capture(
    manifest: dict[str, Any], capture: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, Any]:
    """Annotations must be independently scored, not model-generated guesses."""
    case = Case(**manifest["case"])
    if seal({k: v for k, v in manifest.items() if k != "seal"}) != manifest["seal"]:
        raise ValueError("invalid manifest seal")
    revealed = inspect_capture(case, manifest, capture)
    required = {
        "case_id", "choice_at_E", "probe_prior", "probe_target",
        "probe_relation", "retrieval", "annotator",
    }
    if type(annotation) is not dict or set(annotation) != required:
        raise ValueError("annotation must have exactly the declared fields")
    if annotation["case_id"] != case.case_id or not isinstance(annotation["annotator"], str) or not annotation["annotator"]:
        raise ValueError("annotation identity missing or mismatched")
    choice = _parse_choice(annotation["choice_at_E"], "E classification")
    prior = _parse_choice(annotation["probe_prior"], "probe prior")
    target = _parse_choice(annotation["probe_target"], "probe target")
    try:
        relation = Relation(annotation["probe_relation"])
        retrieval = Retrieval(annotation["retrieval"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid relation or retrieval") from error
    correct_relation = expected_relation(choice, case.target, revealed=revealed)
    correct_target = Choice(case.target) if revealed else Choice.UNKNOWN
    correct_prior = (
        choice if choice in (Choice.A, Choice.B, Choice.CLARIFY) else Choice.UNKNOWN
    )
    prior_ok = prior == correct_prior
    target_ok = target == correct_target
    relation_ok = relation == correct_relation
    return {
        "case_id": case.case_id,
        "lane": case.lane,
        "control": case.control,
        "manifest_seal": manifest["seal"],
        "E_observed_class": choice.value,
        "sealed_target": case.target,  # evaluator-side only
        "E_retrieval_status": retrieval.value,
        "expected_relation": correct_relation.value,
        "prior_correct": prior_ok,
        "target_correct": target_ok,
        "relation_correct": relation_ok,
        "probe_correct": prior_ok and target_ok and relation_ok,
        "classification": (
            "correct" if prior_ok and target_ok and relation_ok
            else "abstain" if prior is Choice.UNKNOWN and correct_prior is not Choice.UNKNOWN
            else "incorrect"
        ),
        "authority_scope": "P0_BENCHMARK_ONLY_NOT_R15",
    }


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def http_chat(base_url: str, message: str) -> dict[str, str]:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query or parsed.fragment
    ):
        raise ValueError("P0 live HTTP is restricted to a localhost origin")
    payload = canonical_bytes({"message": message, "attachments": [], "debug": False})
    req = Request(
        f"http://{parsed.hostname}:{parsed.port}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(_NoRedirect()).open(req, timeout=75) as response:
        obj = json.load(response)
    if (
        type(obj) is not dict or type(obj.get("episode_id")) is not str
        or type(obj.get("response")) is not str
    ):
        raise ValueError("bad ChatResponse")
    return {"episode_id": obj["episode_id"], "response": obj["response"]}


def run_case(
    manifest_path: Path,
    output_path: Path,
    transport: Callable[[str], dict[str, str]],
) -> dict[str, Any]:
    """Run one case in caller-supplied FRESH isolated Suzka instance."""
    case, manifest = read_sealed(manifest_path)
    prompts = case.prompts()
    steps = ("setup", "E", "probe") if case.control == "no_reveal" else (
        "setup", "E", "reveal", "probe"
    )
    started_path = output_path.with_name(output_path.name + ".started.json")
    if output_path.exists() or started_path.exists():
        raise FileExistsError("trial output exists or was started; no retry of a case")
    # Reserve the trial BEFORE the first externally observable chat mutation.
    # An incomplete trial is not safe to retry against the same instance.
    atomic_private_create(started_path, {
        "case_id": case.case_id,
        "manifest_seal": manifest["seal"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "STARTED_NOT_R15_PROOF",
    })
    turns: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        response = transport(prompts[step])
        if (
            type(response) is not dict
            or type(response.get("episode_id")) is not str
            or not response["episode_id"]
            or type(response.get("response")) is not str
        ):
            raise ValueError("invalid transport response")
        turns.append({
            "step": step, "ordinal": i, "request": prompts[step],
            "episode_id": response["episode_id"], "response": response["response"],
        })
    result = {"case_id": case.case_id, "manifest_seal": manifest["seal"], "turns": turns}
    inspect_capture(case, manifest, result)
    atomic_private_create(output_path, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="TEST-ONLY P0 observable reclassification")
    sub = ap.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("cases")
    ls.add_argument("--as-json", action="store_true")
    seal_cmd = sub.add_parser("seal")
    seal_cmd.add_argument("case_id")
    seal_cmd.add_argument("manifest", type=Path)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("manifest", type=Path)
    run_cmd.add_argument("capture", type=Path)
    run_cmd.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_cmd.add_argument("--isolated-instance", action="store_true")
    score_cmd = sub.add_parser("score")
    score_cmd.add_argument("manifest", type=Path)
    score_cmd.add_argument("capture", type=Path)
    score_cmd.add_argument("annotation", type=Path)
    args = ap.parse_args()
    if args.command == "cases":
        cases = [asdict(c) for c in pilot_cases()]
        print(json.dumps(cases if args.as_json else [x["case_id"] for x in cases], indent=2))
    elif args.command == "seal":
        case = next((c for c in pilot_cases() if c.case_id == args.case_id), None)
        if case is None:
            ap.error("unknown predefined synthetic case")
        print(seal_case(case, args.manifest)["seal"])
    elif args.command == "run":
        if not args.isolated_instance:
            ap.error("P0 requires a fresh isolated Suzka instance per case")
        result = run_case(args.manifest, args.capture,
                          lambda msg: http_chat(args.base_url, msg))
        print(json.dumps({"case_id": result["case_id"], "turns": len(result["turns"]),
                          "authority_scope": "P0_BENCHMARK_ONLY_NOT_R15"}))
    elif args.command == "score":
        _, manifest = read_sealed(args.manifest)
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
        print(json.dumps(score_capture(manifest, capture, annotation), indent=2))


if __name__ == "__main__":
    main()
