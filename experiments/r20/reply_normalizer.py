"""COMPROMISED, experiment-side-only normalisation of the S1 / S2 model replies (HD-23, a').

A real model wrapped its S1 reply in a code fence and wrote free text into ``evidence_steps``;
the strict experiment parser (``experiments.r15_p1.protocol.parse_candidate``) refused it.  The
Human accepted a deliberate compromise: this harness (never ``suzka/``, never ``r15_p1``) strips a
whole-reply code fence and replaces ``evidence_steps`` with the fixed expected list.  The model's
own evidence enumeration is therefore NOT measured.  Every other key is left untouched, and a reply
that does not already have the exact closed schema is returned UNCHANGED so the P1 refusal fires.

Only the names of the applied normalisations are ever recorded, never text.
"""

from __future__ import annotations

import json
import re
from typing import Any

FENCE_STRIPPED = "fence_stripped"
EVIDENCE_STEPS_REPLACED = "evidence_steps_replaced"
NAMES = (FENCE_STRIPPED, EVIDENCE_STEPS_REPLACED)
REQUIRED_KEYS = frozenset(
    {"schema", "phase", "assumed_referent", "assessment", "uncertainty", "evidence_steps", "rationale"}
)
# S1 = third transport reply; S2 = fifth (the harness always seals a revealing case: has_reveal).
PHASE_BY_ORDINAL = {3: "S1", 5: "S2"}
EXPECTED_STEPS = {"S1": ["E"], "S2": ["E", "S1", "E2"]}
_FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\n(.*?)\n?```\s*\Z", re.DOTALL)


def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(items)


def normalize_reply(text: str, phase: str | None) -> tuple[str, list[str]]:
    """Return ``(text_for_run_trial, applied_normalisation_names)``."""

    if phase not in EXPECTED_STEPS:
        return text, []
    applied: list[str] = []
    body = text
    match = _FENCE.match(text)
    if match:
        body = match.group(1)
        applied.append(FENCE_STRIPPED)
    try:
        data = json.loads(body, parse_constant=_reject_constant, object_pairs_hook=_pairs)
    except (ValueError, RecursionError):
        return text, []
    if type(data) is not dict or set(data) != REQUIRED_KEYS:
        return text, []
    expected = EXPECTED_STEPS[phase]
    if data["evidence_steps"] != expected:
        data["evidence_steps"] = list(expected)
        applied.append(EVIDENCE_STEPS_REPLACED)
    if not applied:
        return text, []
    return json.dumps(data, ensure_ascii=False), applied


__all__ = ["EVIDENCE_STEPS_REPLACED", "FENCE_STRIPPED", "NAMES", "PHASE_BY_ORDINAL", "normalize_reply"]
