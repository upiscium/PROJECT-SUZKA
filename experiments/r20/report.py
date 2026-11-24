"""The Human-readable report of a real run (IP-23a).

Built from the pre-registration, the ledger, the per-launch result files and the
closed-key manifests only (identifiers, digests, rows, counts, timings, labels;
no raw text).  Defect labels follow #370 / #364:

* ``GATE_CONFORMANCE_DEFECT``: an F1 member whose evaluated row differs from the
  expected table;
* ``ENGINEERING_DEFECT``: a sealed input that does not replay to its stored row;
* ``COGNITIVE_VARIATION``: NEVER assigned by this code (it needs the Human's reading of
  model-owned cognition in a new qualified event);
* ``UNDETERMINED``: any other mismatch or stop: the evidence does not decide.

The report never states a result for R20: the stage judgement is the Human's.
S0 is reported as ``reached`` or ``BLOCKED``, F1 as one of ``ARCHITECTURAL_PATH_OBSERVED`` /
``PARTIAL_VALID_EVIDENCE`` / ``NOT_YET_TESTABLE``.  ``write_report`` refuses a text
that would contain the word for a passed R20.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
from typing import Any

from experiments.r20.caps import Ledger
from experiments.r20.guards import GuardRefusal
from experiments.r20.prereg import load

LABELS = ("GATE_CONFORMANCE_DEFECT", "ENGINEERING_DEFECT", "COGNITIVE_VARIATION", "UNDETERMINED")
_FORBIDDEN = re.compile(r"\bPASS(ED)?\b|(?i:r20[^.\n]{0,24}\bpass(ed)?\b)")
STAGE_JUDGEMENT = "The R20 Stage judgement is the Human's. This report states no R20 result."


def results_files(base: Path) -> list[Path]:
    return sorted(Path(base).glob("results-*.json"))


def manifests(base: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(Path(base).glob("*/evidence/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            found.append(data)
    return found


def defect_labels(results: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Deterministic labels only; everything else is ``UNDETERMINED`` with its reason."""

    labels: dict[str, list[str]] = {name: [] for name in LABELS}
    for result in results:
        for pair in (result.get("f1") or {}).get("pairs", []):
            for member in pair.get("members", []):
                if member.get("defect"):
                    labels[str(member["defect"])].append(f"F1 pair {pair['pair']} {member['arm']}: {member.get('sealed_row')}")
                if member.get("infra") or member.get("stop_reason"):
                    labels["UNDETERMINED"].append(f"F1 pair {pair['pair']} {member['arm']}: {member.get('infra') or member.get('stop_reason')}")
        for attempt in (result.get("s0") or {}).get("attempts", []):
            kill = attempt.get("kill") or {}
            if kill.get("checks", {}).get("replay_of_every_sealed_input_equal") is False:
                labels["ENGINEERING_DEFECT"].append(f"S0 attempt {attempt['attempt']}: a sealed input does not replay")
            counted = attempt.get("step9_structural") or {}
            if counted and not counted.get("equal"):
                labels["UNDETERMINED"].append(f"S0 attempt {attempt['attempt']}: step 9 counts {counted.get('differing')}")
            if attempt.get("stop_reason"):
                labels["UNDETERMINED"].append(f"S0 attempt {attempt['attempt']}: stopped {attempt['stop_reason']}")
    return labels


def _kv_lines(known: dict[str, Any]) -> list[str]:
    lines = ["F-matrix status of F01-F15 (COVERED / TRANSITIONAL / PENDING):"]
    for case in known.get("f_matrix", []):
        note = case.get("transitional") or case.get("pending_reason") or ""
        lines.append(f"- {case['case']} {case['status']}: {case['title']}" + (f" ({note[:160]})" if note else ""))
    lines.append("")
    lines.append("NOT_COVERED_KNOWN_VIOLATION (F15 slice, with the D1 text):")
    for item in known.get("f15_known_violations", []):
        lines.append(f"- {item['id']} {item['title']}: {item['text']}")
    lines.append("- the permanent core-store faults still need the Human.")
    lines.append("")
    lines.append(f"F14 transitional: {known.get('f14_transitional')}")
    lines.append(f"F12 partial coverage: {known.get('f12_partial_coverage')}")
    lines.append("")
    lines.append("Four unlisted known violations (named by the Human before run 1):")
    for index, text in enumerate(known.get("four_unlisted_known_violations", []), 1):
        lines.append(f"{index}. {text}")
    return lines


def build_report(base: Path) -> str:
    content, digest = load(base)
    ledger = Ledger(base)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in results_files(base)]
    lines: list[str] = [f"# R20 real-run report (pre-registration {digest})", ""]
    lines += [
        STAGE_JUDGEMENT,
        "",
        "## What was planned and what ran",
        f"- git head (pre-registered): {content['git_head']}",
        f"- model (pre-registered): {content['model'].get('model_id')} pinned revision {content['model'].get('revision_pinned')}",
        f"- caps (pre-registered): {json.dumps(content['caps'], sort_keys=True)}",
        f"- ledger: {ledger.subjects_used()} subjects started, {ledger.seconds_used() / 3600:.2f} h used",
        f"- result files: {', '.join(path.name for path in results_files(base)) or 'none'}",
        "",
        "## Runs (from the manifests; approval actor HUMAN means the Human approved in the browser)",
    ]
    for manifest in manifests(base):
        real = manifest.get("real_run") or {}
        lines.append(
            f"- {manifest['run_id']} profile={manifest.get('profile')} fixture_only={manifest.get('fixture_only')} "
            f"approval_actor={manifest.get('approval_actor')} admission_path={manifest.get('admission_path')} "
            f"prereg={str(manifest.get('preregistration_sha256'))[:12]} load_s={real.get('serving_after_s')}"
        )
    for result in results:
        stage = result.get("stage")
        lines += ["", f"## Stage {stage} (ended: {result.get('stop_reason')})"]
        if stage == "G1":
            gate = result.get("g1") or {}
            lines.append(f"- G1 label: {gate['label']} (X not skipped as neutral: {gate.get('x_not_skipped_as_neutral')})")
            lines.append(f"- model load s: {gate.get('serving_after_s')}; GPU: {json.dumps(gate.get('gpu'), sort_keys=True)}")
            lines.append(f"- turn seconds (max {gate.get('max_turn_seconds')}): {gate.get('turn_seconds')}")
        if stage == "S0":
            block = result.get("s0") or {"s0": "NOT_COMPLETED", "attempts": []}
            lines.append(f"- S0: {block['s0']} (K = 3 rule: {content['s0']['k_rule']})")
            lines.append("| attempt | first row-4 row | interpretation / stage | receipts | S0 candidate | stop |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for attempt in block["attempts"]:
                first = attempt.get("first_row4") or {}
                lines.append(
                    f"| {attempt['attempt']} | {first.get('row')} | {first.get('interpretation')}/{first.get('stage')} | "
                    f"{first.get('receipts')} | {attempt.get('qualifies_as_s0')} | {attempt.get('stop_reason')} |"
                )
                if attempt.get("step9_structural"):
                    lines.append(f"  - step 9 structural counts equal to the pre-registered: {attempt['step9_structural']['equal']} "
                                 f"{attempt['step9_structural']['counts']}")
                if attempt.get("problems"):
                    lines.append(f"  - problems: {attempt['problems']}")
                if attempt.get("ablation"):
                    lines.append(f"  - ablation: {attempt['ablation']}")
                if attempt.get("journal"):
                    lines.append(f"  - journal bytes {attempt['journal'].get('live_bytes')} of soft band start "
                                 f"{attempt['journal'].get('soft_band_start_bytes')}")
            if block.get("negative_variant"):
                lines.append(f"- negative variant: {json.dumps({k: v for k, v in block['negative_variant'].items() if k != 'run_id'}, sort_keys=True)}")
            lines.append("- approval actor labels: S1 / S2 admission path HARNESS_BRIDGE; approval HUMAN (V1: authorized implies MATCH_SELF, "
                         "so MATCH_SELF is not independent evidence the forecast was right)")
        if stage == "S1":
            block = result.get("s1") or {"g2": {}, "s1": "NOT_COMPLETED", "attempts": []}
            lines.append(f"- G2: {json.dumps(block['g2'], sort_keys=True)}")
            lines.append(f"- S1: {block['s1']}; expectation (pre-registered): {content['s1']['expectation']}")
            for attempt in block["attempts"]:
                lines.append(f"- attempt {attempt['attempt']}: {json.dumps({k: v for k, v in attempt.items() if k != 'run_id'}, sort_keys=True)}")
        if stage == "F1":
            summary = (result.get("f1") or {}).get("summary") or {"claim_ceiling": "NOT_COMPLETED", "i_to_s": "-", "e_r15": None, "valid_treatment_pairs": 0, "control_discordance": 0, "control_pairs_valid": 0, "classes": {}}
            lines.append(f"- F1 claim ceiling: {summary['claim_ceiling']}; I_TO_S: {summary['i_to_s']}")
            lines.append(f"- E_R15 {summary['e_r15']} over {summary['valid_treatment_pairs']} valid treatment pairs; "
                         f"control discordance {summary['control_discordance']} of {summary['control_pairs_valid']}; "
                         f"classes {summary['classes']}")
            lines.append("| pair | kind | class | rows (arm: row) |")
            lines.append("| --- | --- | --- | --- |")
            for pair in (result.get("f1") or {}).get("pairs", []):
                rows = ", ".join(f"{m['arm']}: {m.get('sealed_row')}" for m in pair["members"])
                lines.append(f"| {pair['pair']} | {pair['kind']} | {pair['class']} | {rows} |")
    lines += ["", "## Defect labels (deterministic rules only; COGNITIVE_VARIATION is the Human's reading)"]
    labelled = defect_labels(results)
    for name in LABELS:
        lines.append(f"- {name}: {len(labelled[name])}" + ("".join(f"\n  - {item}" for item in labelled[name])))
    lines += ["", "## Known violations (pre-registered; the report cannot be read as full coverage)"]
    lines += _kv_lines(content["known_violations"])
    lines += ["", "## Not claimed", content["not_claimed"]]
    return "\n".join(lines) + "\n"


def write_report(base: Path, out: Path | None = None) -> Path:
    text = build_report(base)
    if _FORBIDDEN.search(text):
        raise GuardRefusal("REPORT_STATES_A_PASS", "the report never states a pass of R20")
    target = out or Path(base) / "report.md"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    return Path(target)


__all__ = ["LABELS", "STAGE_JUDGEMENT", "build_report", "defect_labels", "write_report"]
