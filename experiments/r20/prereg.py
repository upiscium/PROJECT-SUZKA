"""The pre-registration record (IP-23 sec. 8; created once, immutable).

``create`` writes ``<base>/preregistration.json`` with an exclusive create (a
second call refuses), mode 0400, and its SHA-256 beside it
(``preregistration.sha256``, 0400).  ``load`` re-verifies the file against the
recorded hash and refuses a writable or altered file.  Every launch prints the
hash and every manifest stores it, so the record cannot change between the
first run and the report.

It holds, among others: the git head, the model override (id, pinned revision),
the config template digest, the fixtures by digest, the S0 K = 3 rule and
the structural expected counts of step 9, the F1 table text and plan (master
seed, per-pair seeds, orders, retry cap), the S1 expectation, the caps and stop
rules, the known violations the report carries (derived, not typed), and what is not claimed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from experiments.r15_p0.pilot import Case
from experiments.r20 import f_matrix
from experiments.r20.caps import Caps
from experiments.r20.f1_table import (
    F1_CONTROL_PAIRS,
    F1_PREREGISTRATION_TEXT,
    F1_TOTAL_PAIRS,
    F1_TREATMENT_PAIRS,
)
from experiments.r20.guards import GuardRefusal, repo_root
from experiments.r20.model_config import ModelOverride
from experiments.r20.normalizing_provider import E_ATTEMPTS_MAX, REPLY_ATTEMPTS_MAX
from experiments.r20.provider import S0_TARGET

PREREG_NAME = "preregistration.json"
HASH_NAME = "preregistration.sha256"
SCHEMA = 1
K_RULE = (
    "K = 3 (S0): at most 3 attempts; the order and the template are fixed before attempt 1; the first attempt whose "
    "first row-4 Decision has stage S2 and class SETTLED is S0; every attempt is reported with its plan_hold_receipt "
    "id and sealed row when it is not S0; none qualifies => S0 and R20 are BLOCKED; no status substitution and no "
    "extra attempts."
)
# S0 step 9, read against structural counts (HD-23-4 (a)); logical counts do not depend on the text.
EXPECTED_STRUCTURAL_COUNTS: dict[str, int] = {
    "row4_decisions": 1,
    "intents": 1,
    "execution_claims": 1,
    "target_effects": 1,
    "feedback": 1,
    "feedback_admitted_wakes": 1,
    "outbox_messages": 1,
}
NEGATIVE_VARIANT = "MISSING_APPROVAL: the Human does not approve; row 2c, zero claims, target bytes unchanged, subject keeps running"
S1_EXPECTATION = (
    "S1 (#379): a subject-originated Goal (R18 off) -> GOAL_ADMITTED Wake without a user turn -> re-evaluation. The "
    "driver has already issued row 3 at adoption, so the same fingerprint gives EXISTING and there is no ABANDON path "
    "(#408 known limit): expected 'Plan A continues, or a HOLD with a reason', the cycle closed by the D-R19 "
    "resolution; a user request on the occupied X gets the reasoned HOLD SELF_GOAL_OCCUPIES_TARGET. G2: salience "
    ">= 0.5 on the fixture, else S1 is NOT_YET_TESTABLE."
)
S1_FIXTURE = "I finally finished the thing I had worried about for weeks, and it feels wonderful."
S1_SALIENCE_MIN = 0.5
F1_FORMULAS = (
    "per-pair class priority INFRA_FAILED > NOT_ADMITTED > CONFOUNDED > VALID; E_R15 = (valid treatment pairs whose "
    "two branches' sealed rows differ) / (valid treatment pairs); "
    "control-pair discordance = provider nondeterminism only; claim ceiling: ARCHITECTURAL_PATH_OBSERVED only if "
    "all 12 pairs are valid, any missing / confounded pair gives PARTIAL_VALID_EVIDENCE, no discordant valid "
    "treatment pair gives NOT_YET_TESTABLE; I_TO_S stays DESCRIPTIVE_ONLY / CAUSALITY_UNESTABLISHED."
)
NOT_CLAIMED = (
    "No I->S causality; no R20 Stage result (the Stage judgement is the Human's); a real-model transcript is not an "
    "independent replication; same-I controls measure provider nondeterminism only; the R15 S1 / S2 trigger is the "
    "harness (HARNESS_BRIDGE), not a product runtime. "
    "COMPROMISE (HD-23, a'): S1/S2 model replies are normalised at the model-call layer BEFORE the subject stores "
    "them (code-fence stripped; evidence_steps replaced by the fixed expected list); the subject's memory holds the "
    "normalised text, and the model's literal text is kept only in a private 0600 file; the model's own evidence "
    "enumeration is NOT measured; assumed_referent / assessment / uncertainty / rationale are the model's own. "
    "Reports list which normalisations were applied (names only). "
    "COMPROMISE (HD-23, a, retry): an S1/S2 reply that is still not acceptable after that normalisation is regenerated "
    "from the same request, up to 3 attempts in total (reply_attempts_max); the number of attempts per phase is reported "
    "(numbers only); the stored text is the first acceptable attempt (normalised as before), or the last attempt when "
    "none is acceptable (the P1 / R15 refusal then fires); no other repair is applied. "
    "COMPROMISE (HD-23, E retry): the E reply may be regenerated up to 6 times in total (e_reply_attempts_max), "
    "S1/S2 up to 3 (reply_attempts_max); E gets no normalisation; the stored original answer E is the first attempt "
    "whose prior-referent set is exactly one option, so this selects E; attempts per phase are reported (numbers "
    "only); the effect of that selection on S1-versus-E comparisons is NOT claimed away. "
    "NOTE (HD-23, E2 target): the S0 case's announced referent (the E2 target) is fixed to A because the model's "
    "original choice was B in 6 of 6 observed real runs; a run whose original choice is A would be refused by the "
    "production gate (report does not differ from the original); this is a design adaptation to the model's "
    "observed bias, not a measurement."
)
STOP_RULES = (
    "free GPU below the minimum; a turn longer than the turn cap; 3 consecutive INFRA failures; a guard refusal; a "
    "cap reached; Ctrl-C or the stop file. The run records the stop reason; a stopped attempt counts as an attempt."
)
REQUIRED_KEYS = (
    "schema", "created_utc", "git_head", "model", "config_template_sha256", "request_templates_sha256",
    "fixtures", "s0", "s1", "f1", "caps", "stop_rules", "known_violations", "reply_attempts_max", "e_reply_attempts_max", "not_claimed",
)


def _canonical(content: dict[str, Any]) -> bytes:
    return json.dumps(content, sort_keys=True, indent=1, ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def known_violations_section() -> dict[str, Any]:
    """The F-matrix status of every F01-F15, the F15 known violations with their
    D1 text, F14 transitional, and F12 partial coverage."""

    cases = [
        {"case": case.case, "title": case.title, "status": case.status, "caveat": case.caveat,
         "transitional": case.transitional, "pending_reason": case.pending_reason}
        for case in f_matrix.CASES
    ]
    f15 = f_matrix.case_for("F15")
    return {
        "f_matrix": cases,
        "f15_known_violations": [
            {"id": kv.id, "title": kv.title, "text": kv.text, "source": kv.source,
             "label": f_matrix.NOT_COVERED_KNOWN_VIOLATION}
            for kv in f15.known_violations
        ],
        "f14_transitional": f_matrix.case_for("F14").transitional,
        "f12_partial_coverage": (
            "F12 is partial coverage: F12-r on a realistic S0 root may refuse fail-closed (open risk 13 of IP-22); the report says which part "
            "was and was not exercised, and step 9's 'no stale-branch action after a true rollback' is only as "
            "strong as that."
        ),
    }


def derive_f1_plan(master_seed: int, trial_id: str) -> dict[str, Any]:
    """12 pairs (8 treatment + 4 control), their seeds, the counterbalanced
    branch order and the trial order, all derived from the recorded master seed
    and trial id BEFORE the first run (no choice after seeing an output)."""

    if type(master_seed) is not int or not trial_id:
        raise GuardRefusal("F1_PLAN_INPUT", "an integer master seed and a trial id")

    def digest(*parts: object) -> int:
        return int(sha256_hex(f"{master_seed}|{trial_id}|".encode() + "|".join(map(str, parts)).encode()), 16)

    kinds = (
        ["TREATMENT"] * F1_TREATMENT_PAIRS
        + ["CONTROL_NEUTRAL"] * (F1_CONTROL_PAIRS // 2)
        + ["CONTROL_UNCERTAINTY"] * (F1_CONTROL_PAIRS // 2)
    )
    pairs = []
    for index, kind in enumerate(kinds):
        pairs.append(
            {
                "pair": index + 1,
                "kind": kind,
                "seed": digest("seed", index) % (2**31),
                # counterbalanced: odd pairs run the NEUTRAL branch first, even pairs the other
                "first_branch": "NEUTRAL" if index % 2 == 0 else "UNCERTAINTY",
            }
        )
    trial_order = sorted((pair["pair"] for pair in pairs), key=lambda pair_id: digest("order", pair_id))
    return {"master_seed": master_seed, "trial_id": trial_id, "pairs": pairs, "trial_order": trial_order,
            "whole_pair_retry_cap": 3, "total_pairs": F1_TOTAL_PAIRS, "text": F1_PREREGISTRATION_TEXT,
            "formulas": F1_FORMULAS}


def fixtures_digest(fixtures: dict[str, Any]) -> str:
    return sha256_hex(json.dumps(fixtures, sort_keys=True, ensure_ascii=False).encode())


def code_digests() -> dict[str, str]:
    """The digests of the current code that the record pins: the config template, the request
    templates and the fixtures."""

    repository = repo_root()
    return {
        "config_template_sha256": sha256_hex((repository / "config.yaml").read_bytes()),
        "request_templates_sha256": sha256_hex((repository / "suzka" / "r15" / "request_templates.py").read_bytes()),
        "fixtures_sha256": fixtures_digest(fixtures_section()),
    }


def fixtures_section() -> dict[str, Any]:
    case = Case("r20-fixture", S0_TARGET, "forced", 0, "normal")
    prompts = case.prompts()
    return {
        "x_and_e2_prompts_sha256": sha256_hex(json.dumps(prompts, sort_keys=True, ensure_ascii=False).encode()),
        "r15_case": {"target": case.target, "lane": case.lane, "variant": case.variant, "control": case.control},
        "s1_fixture_sha256": sha256_hex(S1_FIXTURE.encode()),
        "s1_salience_min": S1_SALIENCE_MIN,
    }


def build_content(
    *,
    git_head: str,
    model: ModelOverride,
    caps: Caps,
    master_seed: int,
    trial_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    if git_head == "unknown":
        raise GuardRefusal("GIT_HEAD_UNKNOWN", "a pre-registration needs a known git head")
    digests = code_digests()
    return {
        "schema": SCHEMA,
        "created_utc": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head,
        "model": model.record(),
        "config_template_sha256": digests["config_template_sha256"],
        "request_templates_sha256": digests["request_templates_sha256"],
        "fixtures": fixtures_section(),
        "s0": {
            "k_rule": K_RULE,
            "attempts_max": 3,
            "attempt_order": [1, 2, 3],
            "template": "profile S0, NEUTRAL S1 / S2 templates, the fixed fixture, kill point K4 (an external restart), DC-5 mode refuse",
            "kill_point": "K4",
            "expected_structural_counts": dict(EXPECTED_STRUCTURAL_COUNTS),
            "negative_variant": NEGATIVE_VARIANT,
            "g1": "DRV-1 D with the real model on a throwaway S0 subject; X neutral => NOT_YET_TESTABLE, nothing further is spent",
        },
        "s1": {"expectation": S1_EXPECTATION, "fixture": S1_FIXTURE, "attempts_max": 2, "dc_mode": "refuse"},
        "f1": derive_f1_plan(master_seed, trial_id),
        "caps": caps.record(),
        "stop_rules": STOP_RULES,
        "known_violations": known_violations_section(),
        "reply_attempts_max": REPLY_ATTEMPTS_MAX,  # S1 / S2 only
        "e_reply_attempts_max": E_ATTEMPTS_MAX,  # E only
        "not_claimed": NOT_CLAIMED,
    }


def create(base: Path, content: dict[str, Any]) -> str:
    """Create-only: refuses when a record exists; returns the SHA-256."""

    missing = [key for key in REQUIRED_KEYS if key not in content]
    if missing:
        raise GuardRefusal("PREREG_INCOMPLETE", ",".join(missing))
    payload = _canonical(content)
    digest = sha256_hex(payload)
    path = Path(base) / PREREG_NAME
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError as error:
        raise GuardRefusal("PREREG_EXISTS", "the pre-registration is created once and never replaced") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    hash_descriptor = os.open(Path(base) / HASH_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(hash_descriptor, "w", encoding="utf-8") as handle:
        handle.write(digest + "\n")
    return digest


def load(base: Path) -> tuple[dict[str, Any], str]:
    """The record and its hash, verified: the file must hash to the recorded
    value and must not be writable."""

    path = Path(base) / PREREG_NAME
    try:
        payload = path.read_bytes()
        recorded = (Path(base) / HASH_NAME).read_text(encoding="utf-8").strip()
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as error:
        raise GuardRefusal("PREREG_MISSING", str(base)) from error
    digest = sha256_hex(payload)
    if digest != recorded:
        raise GuardRefusal("PREREG_ALTERED", "the file does not hash to the recorded value")
    if mode & 0o222:
        raise GuardRefusal("PREREG_WRITABLE", oct(mode))
    content = json.loads(payload)
    if not isinstance(content, dict) or content.get("schema") != SCHEMA:
        raise GuardRefusal("PREREG_INVALID")
    return content, digest


__all__ = [
    "EXPECTED_STRUCTURAL_COUNTS", "HASH_NAME", "K_RULE", "PREREG_NAME", "S1_FIXTURE", "S1_SALIENCE_MIN",
    "build_content", "create", "derive_f1_plan", "known_violations_section", "load", "sha256_hex",
]
