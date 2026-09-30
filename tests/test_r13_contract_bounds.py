"""Schema-derived R13 size-budget and cross-contract tests."""

from dataclasses import fields
from enum import Enum
import math
import subprocess
import sys

import pytest

from suzka.motivation import (
    R13_SCHEMA_SIZE_BUDGET,
    canonical_r13_schema_size_budget,
    derive_r13_schema_size_budget,
    validate_r13_schema_size_budget,
)
from suzka.runtime.agent_state import (
    AGENT_STATE_FUTURE_STATE_RESERVE_BYTES,
    AGENT_STATE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES,
    project_agent_state_schema_max_bytes,
)
from suzka.motivation.bounds import (
    R13_MAX_FLOAT_HEX,
    R13_MAX_REF,
    R13_SCHEMA_MAX_AGGREGATE_BYTES,
    R13_SCHEMA_MAX_COMMITMENT_RECORD_BYTES,
    R13_SCHEMA_MAX_COMMITMENT_SECTION_BYTES,
    R13_SCHEMA_MAX_GOAL_RECORD_BYTES,
    R13_SCHEMA_MAX_GOAL_SECTION_BYTES,
    R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_BYTES,
    R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_SECTION_BYTES,
    R13_SCHEMA_MAX_MOTIVATION_RECORD_BYTES,
    R13_SCHEMA_MAX_MOTIVATION_SECTION_BYTES,
    max_commitment_record_shape,
    max_goal_record_shape,
    max_motivation_candidate_shape,
    max_motivation_record_shape,
)
from suzka.motivation.commitment import (
    COMMITMENT_BENEFICIARY_KINDS,
    COMMITMENT_EVIDENCE_KINDS,
    COMMITMENT_ORIGIN_KINDS,
    COMMITMENT_OUTCOME_KINDS,
    CommitmentLifecycle,
    CommitmentRecord,
    CommitmentRevisionOperation,
    CommitmentRevisionReason,
)
from suzka.motivation.common import (
    R13_MAX_CANDIDATES_PER_DOMAIN,
    R13_MAX_RECORDS_PER_DOMAIN,
    R13_MAX_SHORT_TEXT_CODEPOINTS,
    R13ReferenceKind,
    R13_SCHEMA_VERSION,
    bounded_fraction,
    canonical_json,
)
from suzka.motivation.goal import (
    GOAL_EVIDENCE_KINDS,
    GOAL_ORIGIN_KINDS,
    GOAL_OUTCOME_KINDS,
    GOAL_TARGET_KINDS,
    GoalLifecycle,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
)
from suzka.motivation.motivation import (
    MOTIVATION_RELATED_KINDS,
    MOTIVATION_SOURCE_KINDS,
    MOTIVATION_TARGET_KINDS,
    MotivationCandidateClassification,
    MotivationKind,
    MotivationLifecycle,
    MotivationRecord,
    MotivationRevisionOperation,
    MotivationRevisionReason,
)


def test_schema_budget_is_reproducible_and_preserves_the_v8_future_reserve() -> None:
    budget = derive_r13_schema_size_budget()
    assert budget == R13_SCHEMA_SIZE_BUDGET
    assert validate_r13_schema_size_budget() == budget
    assert R13_MAX_RECORDS_PER_DOMAIN == 32
    assert R13_MAX_CANDIDATES_PER_DOMAIN == 32
    assert budget.aggregate_r13_bytes == R13_SCHEMA_MAX_AGGREGATE_BYTES
    assert budget.has_provisional_capacity_headroom
    assert budget.agent_state_cap_bytes == AGENT_STATE_MAX_SERIALIZED_BYTES
    assert budget.agent_state_v7_base_bytes == (
        AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES
    )
    assert budget.agent_state_v7_base_bytes == 85_826_260
    assert budget.aggregate_r13_bytes == 18_759_305
    assert budget.projected_agent_state_v8_bytes == 104_585_585
    assert budget.agent_state_future_state_reserve_bytes == (
        AGENT_STATE_FUTURE_STATE_RESERVE_BYTES
    )
    assert budget.projected_agent_state_v8_with_reserve_bytes == (
        project_agent_state_schema_max_bytes(
            schema_version=8,
            added_field_maxima={"motivation_state": budget.aggregate_r13_bytes},
        )
    )
    assert budget.projected_agent_state_v8_with_reserve_bytes == 121_362_801
    assert budget.remaining_after_r13_bytes == 29_632_143
    assert budget.remaining_after_r13_bytes >= (
        budget.agent_state_future_state_reserve_bytes
    )
    assert budget.remaining_after_reserve_bytes == 12_854_927
    assert budget.remaining_after_reserve_bytes >= 0
    assert canonical_r13_schema_size_budget() == canonical_r13_schema_size_budget()


def test_importing_pure_r13_contracts_does_not_import_runtime_or_models() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import suzka.motivation; "
                "assert 'suzka.runtime' not in sys.modules; "
                "assert 'suzka.runtime.agent_state' not in sys.modules; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (max_motivation_record_shape(), R13_SCHEMA_MAX_MOTIVATION_RECORD_BYTES),
        (max_goal_record_shape(), R13_SCHEMA_MAX_GOAL_RECORD_BYTES),
        (max_commitment_record_shape(), R13_SCHEMA_MAX_COMMITMENT_RECORD_BYTES),
        (max_motivation_candidate_shape(), R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_BYTES),
    ],
)
def test_record_budget_is_the_canonical_shape_size(
    shape: dict[str, object], expected: int
) -> None:
    assert len(canonical_json(shape)) == expected


def _assert_longest_reference(
    value: object,
    allowed_kinds: frozenset[R13ReferenceKind],
) -> None:
    assert isinstance(value, dict)
    kind = value.get("kind")
    reference = value.get("reference")
    assert isinstance(kind, str)
    assert kind in {item.value for item in allowed_kinds}
    assert len(kind) == max(len(item.value) for item in allowed_kinds)
    assert isinstance(reference, str)
    assert len(reference) == R13_MAX_REF


def _assert_longest_reference_list(
    shape: dict[str, object],
    field: str,
    allowed_kinds: frozenset[R13ReferenceKind],
) -> None:
    refs = shape[field]
    assert isinstance(refs, list)
    assert refs
    for reference in refs:
        _assert_longest_reference(reference, allowed_kinds)


def test_max_shapes_use_actual_longest_typed_reference_vocabularies() -> None:
    motivation = max_motivation_record_shape()
    _assert_longest_reference_list(
        motivation,
        "conflict_refs",
        MOTIVATION_SOURCE_KINDS | {R13ReferenceKind.STATE},
    )
    _assert_longest_reference_list(
        motivation, "evidence_refs", MOTIVATION_SOURCE_KINDS
    )
    _assert_longest_reference_list(
        motivation, "related_refs", MOTIVATION_RELATED_KINDS
    )
    _assert_longest_reference(
        motivation["source_evidence"], MOTIVATION_SOURCE_KINDS
    )
    _assert_longest_reference(motivation["target"], MOTIVATION_TARGET_KINDS)

    goal = max_goal_record_shape()
    _assert_longest_reference_list(goal, "evidence_refs", GOAL_EVIDENCE_KINDS)
    _assert_longest_reference_list(goal, "origin_refs", GOAL_ORIGIN_KINDS)
    _assert_longest_reference_list(
        goal, "outcome_evidence_refs", GOAL_OUTCOME_KINDS
    )
    _assert_longest_reference(goal["target"], GOAL_TARGET_KINDS)
    assert goal["evidence_refs"][0]["kind"] == "operator_proposal"
    assert goal["outcome_evidence_refs"][0]["kind"] == "experience"

    commitment = max_commitment_record_shape()
    _assert_longest_reference(
        commitment["beneficiary"], COMMITMENT_BENEFICIARY_KINDS
    )
    _assert_longest_reference_list(
        commitment, "evidence_refs", COMMITMENT_EVIDENCE_KINDS
    )
    _assert_longest_reference_list(
        commitment, "origin_refs", COMMITMENT_ORIGIN_KINDS
    )
    _assert_longest_reference_list(
        commitment, "outcome_evidence_refs", COMMITMENT_OUTCOME_KINDS
    )
    _assert_longest_reference_list(
        commitment,
        "related_goal_refs",
        frozenset({R13ReferenceKind.GOAL}),
    )
    _assert_longest_reference_list(
        commitment,
        "desire_refs",
        frozenset({R13ReferenceKind.MOTIVATION}),
    )
    assert commitment["outcome_evidence_refs"][0]["kind"] == "experience"

    candidate = max_motivation_candidate_shape()
    _assert_longest_reference_list(
        candidate, "source_evidence_refs", MOTIVATION_SOURCE_KINDS
    )
    _assert_longest_reference(
        candidate["suggested_target"], MOTIVATION_TARGET_KINDS
    )
    model_identity = candidate["model_identity"]
    assert isinstance(model_identity, dict)
    assert len(model_identity["model_id"]) == R13_MAX_SHORT_TEXT_CODEPOINTS
    assert len(model_identity["provider_name"]) == R13_MAX_SHORT_TEXT_CODEPOINTS


def _assert_longest_scalar(value: object, enum_type: type[Enum]) -> None:
    assert isinstance(value, str)
    values = [str(member.value) for member in enum_type]
    assert value in values
    assert len(value) == max(len(item) for item in values)


def test_max_shapes_use_valid_longest_scalar_and_revision_vocabularies() -> None:
    motivation = max_motivation_record_shape()
    _assert_longest_scalar(motivation["kind"], MotivationKind)
    _assert_longest_scalar(motivation["lifecycle"], MotivationLifecycle)

    goal = max_goal_record_shape()
    _assert_longest_scalar(goal["lifecycle"], GoalLifecycle)

    commitment = max_commitment_record_shape()
    _assert_longest_scalar(commitment["lifecycle"], CommitmentLifecycle)

    candidate = max_motivation_candidate_shape()
    _assert_longest_scalar(
        candidate["classification"], MotivationCandidateClassification
    )
    _assert_longest_scalar(candidate["suggested_kind"], MotivationKind)

    motivation_anchor = motivation["history_anchor"]
    assert isinstance(motivation_anchor, dict)
    assert (
        motivation_anchor["through_state"],
        motivation_anchor["through_previous_state"],
        motivation_anchor["through_operation"],
        motivation_anchor["through_reason"],
    ) == ("active", "satiated", "update", "evidence_update")

    revision_expectations = (
        (
            motivation,
            MotivationRevisionOperation,
            MotivationRevisionReason,
            {
                ("create", "creation"),
                ("update", "evidence_update"),
                ("update", "decay"),
                ("update", "satiation"),
                ("retire", "retirement"),
            },
        ),
        (
            goal,
            GoalRevisionOperation,
            GoalRevisionReason,
            {
                ("create", "creation"),
                ("adopt", "subject_admission"),
                ("defer", "subject_deferred"),
                ("abandon", "subject_abandoned"),
                ("complete", "verified_outcome"),
                ("fail", "verified_outcome"),
            },
        ),
        (
            commitment,
            CommitmentRevisionOperation,
            CommitmentRevisionReason,
            {
                ("create", "creation"),
                ("admit", "subject_admission"),
                ("release", "subject_release"),
                ("renegotiate", "subject_renegotiation"),
                ("fulfill", "verified_outcome"),
                ("breach", "verified_outcome"),
            },
        ),
    )
    for shape, operation_enum, reason_enum, valid_pairs in revision_expectations:
        history = shape["revision_history"]
        assert isinstance(history, list)
        assert history
        revision = history[0]
        assert isinstance(revision, dict)
        operation = revision["operation"]
        reason = revision["reason"]
        assert isinstance(operation, str)
        assert isinstance(reason, str)
        assert operation in {item.value for item in operation_enum}
        assert reason in {item.value for item in reason_enum}
        assert (operation, reason) in valid_pairs
        assert len(operation) + len(reason) == max(
            len(item[0]) + len(item[1]) for item in valid_pairs
        )


def _replace_reference_kinds(refs: object, kind: str) -> list[dict[str, str]]:
    assert isinstance(refs, list)
    result: list[dict[str, str]] = []
    for reference in refs:
        assert isinstance(reference, dict)
        identifier = reference.get("reference")
        assert isinstance(identifier, str)
        result.append({"kind": kind, "reference": identifier})
    return result


def _section(shape: dict[str, object], count: int) -> dict[str, object]:
    return {
        "authority_digest": "d" * 64,
        "records": [dict(shape) for _ in range(count)],
        "schema_version": R13_SCHEMA_VERSION,
    }


def test_independent_longest_witnesses_fit_record_section_and_aggregate_budgets(
) -> None:
    motivation = max_motivation_record_shape()
    goal = max_goal_record_shape()
    commitment = max_commitment_record_shape()
    candidate = max_motivation_candidate_shape()

    longest_goal_evidence = max(
        (item.value for item in GOAL_EVIDENCE_KINDS),
        key=lambda value: (len(value), value),
    )
    longest_goal_outcome = max(
        (item.value for item in GOAL_OUTCOME_KINDS),
        key=lambda value: (len(value), value),
    )
    goal["evidence_refs"] = _replace_reference_kinds(
        goal["evidence_refs"], longest_goal_evidence
    )
    goal["outcome_evidence_refs"] = _replace_reference_kinds(
        goal["outcome_evidence_refs"], longest_goal_outcome
    )

    longest_commitment_outcome = max(
        (item.value for item in COMMITMENT_OUTCOME_KINDS),
        key=lambda value: (len(value), value),
    )
    commitment["outcome_evidence_refs"] = _replace_reference_kinds(
        commitment["outcome_evidence_refs"], longest_commitment_outcome
    )
    for revision in commitment["revision_history"]:
        assert isinstance(revision, dict)
        revision["operation"] = "renegotiate"
        revision["reason"] = "subject_renegotiation"

    assert len(canonical_json(motivation["revision_history"][0])) <= (
        R13_SCHEMA_SIZE_BUDGET.motivation_revision_bytes
    )
    assert len(canonical_json(goal["revision_history"][0])) <= (
        R13_SCHEMA_SIZE_BUDGET.goal_revision_bytes
    )
    assert len(canonical_json(commitment["revision_history"][0])) <= (
        R13_SCHEMA_SIZE_BUDGET.commitment_revision_bytes
    )
    assert (
        len(canonical_json(motivation))
        <= R13_SCHEMA_MAX_MOTIVATION_RECORD_BYTES
    )
    assert len(canonical_json(goal)) <= R13_SCHEMA_MAX_GOAL_RECORD_BYTES
    assert (
        len(canonical_json(commitment))
        <= R13_SCHEMA_MAX_COMMITMENT_RECORD_BYTES
    )
    assert (
        len(canonical_json(candidate))
        <= R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_BYTES
    )

    motivation_section = _section(motivation, R13_MAX_RECORDS_PER_DOMAIN)
    goal_section = _section(goal, R13_MAX_RECORDS_PER_DOMAIN)
    commitment_section = _section(commitment, R13_MAX_RECORDS_PER_DOMAIN)
    candidate_section = _section(candidate, R13_MAX_CANDIDATES_PER_DOMAIN)
    assert (
        len(canonical_json(motivation_section))
        <= R13_SCHEMA_MAX_MOTIVATION_SECTION_BYTES
    )
    assert len(canonical_json(goal_section)) <= R13_SCHEMA_MAX_GOAL_SECTION_BYTES
    assert (
        len(canonical_json(commitment_section))
        <= R13_SCHEMA_MAX_COMMITMENT_SECTION_BYTES
    )
    assert (
        len(canonical_json(candidate_section))
        <= R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_SECTION_BYTES
    )

    aggregate = canonical_json(
        {
            "commitment": commitment_section,
            "goal": goal_section,
            "motivation": motivation_section,
            "motivation_candidates": candidate_section,
            "schema_version": R13_SCHEMA_VERSION,
        }
    )
    assert len(aggregate) <= R13_SCHEMA_MAX_AGGREGATE_BYTES


def test_fraction_budget_covers_smallest_positive_subnormal() -> None:
    smallest_positive = math.nextafter(0.0, 1.0)

    assert bounded_fraction(smallest_positive, "fraction") == smallest_positive
    assert R13_MAX_FLOAT_HEX == smallest_positive.hex()
    assert len(R13_MAX_FLOAT_HEX) == 23


def test_u1_contracts_have_no_private_or_mutable_authority_surface() -> None:
    for contract in (MotivationRecord, GoalRecord, CommitmentRecord):
        names = {item.name.casefold() for item in fields(contract)}
        assert not names.intersection(
            {
                "prompt",
                "raw_prompt",
                "transcript",
                "hidden_thought",
                "private_reasoning",
                "model_rationale",
            }
        )
        assert not names.intersection({"store", "runtime", "scheduler", "outbox"})
