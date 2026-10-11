"""Test-only evidence projection and fail-closed P0 answer checker.

These are evaluator assertions over synthetic scenarios, not R07/R12/R09
source authentication or R15 first-person semantic admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Any

from experiments.r15_p0.pilot import (
    Case,
    Choice,
    Relation,
    canonical_bytes,
    expected_relation,
    inspect_capture,
    seal,
)


class SelectionStatus(StrEnum):
    INCLUDED = "included"
    OVER_BUDGET = "over_budget"


class GateStatus(StrEnum):
    TEST_ONLY = "test_only"
    EVIDENCE_NOT_PROJECTED = "evidence_not_projected"
    MALFORMED_MODEL_CLAIM = "malformed_model_claim"
    UNVERIFIED_E_ANNOTATION = "unverified_e_annotation"


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    case_id: str
    manifest_seal: str
    entries: tuple[tuple[str, str, str, str | None], ...]
    has_reveal: bool
    source_bytes: int

    def source_value(self) -> dict[str, Any]:
        return {
            "authority": "P0_BENCHMARK_OBSERVATION_ONLY",
            "case_id": self.case_id,
            "reveal_state": "PRESENT" if self.has_reveal else "ABSENT",
            "sources": [
                {
                    "step": step,
                    "episode_id": episode_id,
                    "user_input": user_input,
                    "visible_answer": answer,
                }
                for step, episode_id, user_input, answer in self.entries
            ],
        }


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    status: SelectionStatus
    packet: EvidencePacket | None
    required_bytes: int
    budget_bytes: int

    def __post_init__(self) -> None:
        if type(self.budget_bytes) is not int or not 256 <= self.budget_bytes <= 8192:
            raise ValueError("invalid P0 evidence budget")
        if self.required_bytes < 0:
            raise ValueError("negative source size")
        if self.status is SelectionStatus.INCLUDED:
            if self.packet is None or self.required_bytes > self.budget_bytes:
                raise ValueError("included source must fit exactly")
        elif self.status is SelectionStatus.OVER_BUDGET:
            if self.packet is not None or self.required_bytes <= self.budget_bytes:
                raise ValueError("over-budget source cannot be partially published")
        else:
            raise ValueError("unknown source selection status")


@dataclass(frozen=True, slots=True)
class ModelTriplet:
    original_choice: Choice
    revealed_target: Choice
    relation: Relation

    def __post_init__(self) -> None:
        if self.original_choice not in (Choice.A, Choice.B, Choice.CLARIFY, Choice.UNKNOWN):
            raise ValueError("invalid original-choice claim")
        if self.revealed_target not in (Choice.A, Choice.B, Choice.UNKNOWN):
            raise ValueError("invalid revealed-target claim")


UNKNOWN_TRIPLET = ModelTriplet(Choice.UNKNOWN, Choice.UNKNOWN, Relation.UNKNOWN)


@dataclass(frozen=True, slots=True)
class GateDecision:
    status: GateStatus
    raw: ModelTriplet | None
    admitted: ModelTriplet
    checks: tuple[tuple[str, bool], ...]
    rejected: tuple[str, ...]
    authority: str = "P0_TEST_EVALUATOR_NOT_SUZKA_R15"

    @property
    def all_three_correct(self) -> bool:
        return (
            self.raw is not None
            and all(passed for _, passed in self.checks)
            and not self.rejected
            and self.status is GateStatus.TEST_ONLY
        )


def _checked_case(manifest: dict[str, Any]) -> Case:
    if type(manifest) is not dict or set(manifest) != {
        "schema", "case", "created_at", "seal"
    }:
        raise ValueError("invalid closed P0 manifest")
    if type(manifest["schema"]) is not int or manifest["schema"] != 1:
        raise ValueError("invalid manifest schema")
    if type(manifest["created_at"]) is not str or not manifest["created_at"]:
        raise ValueError("missing manifest time")
    if type(manifest["seal"]) is not str or type(manifest["case"]) is not dict:
        raise ValueError("invalid manifest structure")
    if set(manifest["case"]) != {"case_id", "target", "lane", "variant", "control"}:
        raise ValueError("invalid case fields")
    case = Case(**manifest["case"])
    if seal({k: v for k, v in manifest.items() if k != "seal"}) != manifest["seal"]:
        raise ValueError("manifest digest mismatch")
    return case


def select_evidence(
    manifest: dict[str, Any],
    capture: dict[str, Any],
    *,
    budget_bytes: int = 2048,
) -> EvidenceSelection:
    """Select whole evidence excerpts or none, never silently clip one source.

    Includes scenario setup request, E question+visible answer, and, only if
    captured, the actual later reveal *user request*. Omits unrelated model
    response to the reveal. This is a test-side projection, not source trust.
    """
    if type(budget_bytes) is not int or not 256 <= budget_bytes <= 8192:
        raise ValueError("invalid P0 source budget")
    case = _checked_case(manifest)
    inspect_capture(case, manifest, capture)
    by_step = {r["step"]: r for r in capture["turns"]}
    has_reveal = case.control != "no_reveal"
    entries: tuple[tuple[str, str, str, str | None], ...] = (
        ("setup", by_step["setup"]["episode_id"], by_step["setup"]["request"], None),
        ("E", by_step["E"]["episode_id"], by_step["E"]["request"], by_step["E"]["response"]),
        *(
            (("E2", by_step["reveal"]["episode_id"], by_step["reveal"]["request"], None),)
            if has_reveal else ()
        ),
    )
    candidate = EvidencePacket(case.case_id, manifest["seal"], entries, has_reveal, 0)
    required = len(canonical_bytes(candidate.source_value()))
    if required > budget_bytes:
        return EvidenceSelection(SelectionStatus.OVER_BUDGET, None, required, budget_bytes)
    return EvidenceSelection(
        SelectionStatus.INCLUDED,
        EvidencePacket(case.case_id, manifest["seal"], entries, has_reveal, required),
        required, budget_bytes,
    )


def model_question(selection: EvidenceSelection) -> str:
    """Render benchmark-only source text, never a production Attention rewrite."""
    if selection.status is not SelectionStatus.INCLUDED or selection.packet is None:
        raise ValueError("unavailable source must not be sent to a model")
    packet = json.dumps(selection.packet.source_value(), sort_keys=True, ensure_ascii=False)
    return (
        "これはテスト用の観測ログです。現在の指示で過去の出来事を作らないでください。"
        "reveal_stateがABSENTの場合はrevealed_targetとrelationをUNKNOWNにしてください。"
        "過去にA/Bのどちらも明示的に選んでいなければoriginal_choice=UNKNOWNです。"
        "証拠で判断できないものはUNKNOWNです。厳密に次の3行だけ英大文字で回答してください。\n"
        "original_choice: A/B/CLARIFY/UNKNOWN\n"
        "revealed_target: A/B/UNKNOWN\n"
        "relation: MATCH/MISMATCH/NO_CHOICE/UNKNOWN\n"
        "観測された資料:\n" + packet
    )


def parse_triplet(text: str) -> ModelTriplet:
    if type(text) is not str:
        raise ValueError("model response must be a string")
    rows = text.strip().splitlines()
    prefixes = ("original_choice: ", "revealed_target: ", "relation: ")
    if len(rows) != 3 or any(
        not row.startswith(prefix) for row, prefix in zip(rows, prefixes, strict=True)
    ):
        raise ValueError("invalid P0 answer keys or number of lines")
    v = tuple(row[len(key):] for row, key in zip(rows, prefixes, strict=True))
    if (
        v[0] not in {"A", "B", "CLARIFY", "UNKNOWN"}
        or v[1] not in {"A", "B", "UNKNOWN"}
        or v[2] not in {r.value for r in Relation}
    ):
        raise ValueError("invalid P0 answer enum")
    return ModelTriplet(Choice(v[0]), Choice(v[1]), Relation(v[2]))


def gate_model_claim(
    manifest: dict[str, Any],
    capture: dict[str, Any],
    selected: EvidenceSelection,
    response: str | None,
    *,
    blind_e_choice: Choice | None = None,
    independent_e_annotation: bool = False,
) -> GateDecision:
    """Admit no unsupported target, identity, or relation from a model answer.

    'independent_e_annotation' is an evaluator assertion, NOT authentication.
    This is deliberately NOT usable as an R15 subject-admission primitive.
    """
    current = select_evidence(manifest, capture, budget_bytes=selected.budget_bytes)
    if selected != current:
        raise ValueError("stale or forged test source projection")
    if selected.status is SelectionStatus.OVER_BUDGET:
        if response is not None:
            raise ValueError("cannot score against unavailable source")
        return GateDecision(
            GateStatus.EVIDENCE_NOT_PROJECTED, None, UNKNOWN_TRIPLET,
            (("original_choice", False), ("revealed_target", False), ("relation", False)),
            ("source_over_budget",),
        )
    if response is None:
        return GateDecision(
            GateStatus.MALFORMED_MODEL_CLAIM, None, UNKNOWN_TRIPLET,
            (("original_choice", False), ("revealed_target", False), ("relation", False)),
            ("missing_model_response",),
        )
    try:
        raw = parse_triplet(response)
    except ValueError:
        return GateDecision(
            GateStatus.MALFORMED_MODEL_CLAIM, None, UNKNOWN_TRIPLET,
            (("original_choice", False), ("revealed_target", False), ("relation", False)),
            ("unparseable_model_response",),
        )
    case = _checked_case(manifest)
    assert selected.packet is not None
    has_reveal = selected.packet.has_reveal
    labeled = (
        type(independent_e_annotation) is bool
        and independent_e_annotation
        and type(blind_e_choice) is Choice
    )
    prior = (
        blind_e_choice
        if labeled and blind_e_choice in (Choice.A, Choice.B, Choice.CLARIFY)
        else Choice.UNKNOWN
    )
    target = Choice(case.target) if has_reveal else Choice.UNKNOWN
    relation = (
        expected_relation(prior, case.target, revealed=True)
        if has_reveal and prior is not Choice.UNKNOWN else Relation.UNKNOWN
    )
    prior_ok = raw.original_choice is prior
    target_ok = raw.revealed_target is target
    relation_ok = raw.relation is relation and prior_ok and target_ok
    admitted = ModelTriplet(
        raw.original_choice if prior_ok else Choice.UNKNOWN,
        raw.revealed_target if target_ok else Choice.UNKNOWN,
        raw.relation if relation_ok else Relation.UNKNOWN,
    )
    checks = (
        ("original_choice", prior_ok),
        ("revealed_target", target_ok),
        ("relation", relation_ok),
    )
    return GateDecision(
        GateStatus.TEST_ONLY if labeled else GateStatus.UNVERIFIED_E_ANNOTATION,
        raw, admitted, checks,
        tuple("unsupported_" + field for field, ok in checks if not ok),
    )
