"""Source-only U1 regression tests: shape/digests never confer authority."""

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
from typing import cast

import pytest

import suzka.r16 as r16
from suzka.r16 import contracts
from suzka.r16.contracts import (
    MAX_CANDIDATES,
    MAX_EDGES,
    MAX_JSON_BYTES,
    MAX_STEPS,
    SCHEMA_VERSION,
    UNCERTAINTY_SCALE,
    CandidateKind,
    CandidateSelection,
    DecisionRecord,
    ExplanationCategory,
    ExplanationProjection,
    ExplanationStatus,
    GoalEvidenceRef,
    HoldAssessment,
    PlanCandidate,
    PlanDAG,
    PlanStep,
    ReasonCode,
    StepKind,
    ValueEvidenceRef,
)

GOAL_ID = "a" * 64
GOAL_DIGEST = "b" * 64
VALUE_ID = "private-value-17"
VALUE_DIGEST = "c" * 64
PLAN_DOMAIN = b"PROJECT-SUZKA:R16:PLAN-DAG:V1\0"
DECISION_DOMAIN = b"PROJECT-SUZKA:R16:DECISION-RECORD:V1\0"
EXPLANATION_DOMAIN = b"PROJECT-SUZKA:R16:EXPLANATION-PROJECTION:V1\0"


def canonical(value: object) -> bytes:
    """Independent spelling of the wire contract, not the production codec."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def seal(value: dict[str, object], domain: bytes) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "content_digest"}
    checksum = hashlib.sha256(domain + canonical(body)).hexdigest()
    return {**body, "content_digest": checksum}


def plan(index: int = 0) -> PlanDAG:
    return PlanDAG(
        1,
        f"private-plan-{index}",
        (
            PlanStep(1, "assess", StepKind.ASSESS, ()),
            PlanStep(
                1,
                "review",
                StepKind.CHECK if index == 0 else StepKind.COMPARE,
                ("assess",),
            ),
        ),
    )


def candidates() -> tuple[PlanCandidate, ...]:
    goal = GoalEvidenceRef(1, GOAL_ID, 2, GOAL_DIGEST)
    value = ValueEvidenceRef(1, VALUE_ID, 3, VALUE_DIGEST)
    return (
        PlanCandidate(1, "candidate-a", CandidateKind.PLAN, goal, value, plan()),
        PlanCandidate(1, "candidate-b", CandidateKind.PLAN, goal, value, plan(1)),
        PlanCandidate(1, "candidate-noop", CandidateKind.NO_OP, goal, value, None),
    )


def decision(*, no_op: bool = False) -> DecisionRecord:
    options = candidates()
    chosen = options[2] if no_op else options[0]
    return DecisionRecord(
        1,
        "private-decision-1",
        options,
        CandidateSelection(1, chosen.candidate_id, chosen.content_digest),
        ReasonCode.NO_OP_PREFERRED if no_op else ReasonCode.PLAN_PREFERRED,
        250_000,
    )


def test_all_pure_shapes_have_exact_canonical_roundtrips_and_are_frozen() -> None:
    record = decision()
    hold = HoldAssessment(1, ReasonCode.INSUFFICIENT_EVIDENCE, UNCERTAINTY_SCALE)
    values = (
        record.candidates[0].goal,
        record.candidates[0].value,
        plan().steps[0],
        plan(),
        *record.candidates,
        record.selected,
        record,
        decision(no_op=True),
        hold,
        record.explanation(),
        hold.explanation(),
    )
    for value in values:
        raw = value.canonical_value()
        wire = value.canonical_bytes()
        assert wire == canonical(raw)
        assert type(value).from_value(raw) == value
        assert type(value).from_json(wire) == value
        assert type(value).from_json(wire).canonical_bytes() == wire
        with pytest.raises(FrozenInstanceError):
            setattr(value, "schema_version", 2)
        with pytest.raises((TypeError, AttributeError)):
            setattr(value, "injected_text", "private")


def test_valid_two_step_plans_compete_with_an_explicit_selectable_no_op() -> None:
    assert (MAX_STEPS, MAX_EDGES, MAX_CANDIDATES, SCHEMA_VERSION) == (4, 6, 3, 1)
    record = decision()
    assert len(record.candidates) == 3
    assert sum(item.kind is CandidateKind.PLAN for item in record.candidates) == 2
    assert record.candidates[2].kind is CandidateKind.NO_OP
    assert record.candidates[2].plan is None
    assert record.selected.candidate_digest == record.candidates[0].content_digest
    noop = decision(no_op=True)
    assert noop.selected.candidate_id == noop.candidates[2].candidate_id
    assert noop.explanation().category is ExplanationCategory.NO_OP
    assert noop.explanation().status is ExplanationStatus.SELECTED


def test_digest_binds_exact_fields_and_is_sha256_domain_separated() -> None:
    value = plan().canonical_value()
    assert value == seal(value, PLAN_DOMAIN)
    body = {key: item for key, item in value.items() if key != "content_digest"}
    assert value["content_digest"] != hashlib.sha256(canonical(body)).hexdigest()
    wrong_domain = seal(value, DECISION_DOMAIN)
    with pytest.raises(ValueError, match="digest"):
        PlanDAG.from_value(wrong_domain)
    changed = dict(value, plan_id="changed-plan")
    with pytest.raises(ValueError, match="digest"):
        PlanDAG.from_value(changed)
    assert PlanDAG.from_value(seal(changed, PLAN_DOMAIN)).plan_id == "changed-plan"
    changed_plan = replace(plan(), plan_id="changed-plan")
    assert changed_plan.content_digest != plan().content_digest


def test_no_mutation_or_aliasing_of_nested_canonical_values() -> None:
    record = decision()
    before = record.canonical_bytes()
    raw = record.canonical_value()
    options = cast(list[dict[str, object]], raw["candidates"])
    nested_plan = cast(dict[str, object], options[0]["plan"])
    steps = cast(list[dict[str, object]], nested_plan["steps"])
    deps = cast(list[str], steps[1]["depends_on"])
    deps.append("injected")
    options[0]["candidate_id"] = "injected"
    assert record.canonical_bytes() == before
    assert plan().topological_order == ("assess", "review")
    with pytest.raises((TypeError, ValueError)):
        replace(plan(), steps=list(plan().steps))
    with pytest.raises((TypeError, ValueError)):
        replace(plan().steps[1], depends_on=["assess"])
    with pytest.raises((TypeError, ValueError)):
        replace(record, candidates=list(record.candidates))


def test_topological_order_uses_least_currently_ready_id_without_mutation() -> None:
    steps = (
        PlanStep(1, "b", StepKind.ASSESS, ()),
        PlanStep(1, "a", StepKind.CHECK, ("b",)),
        PlanStep(1, "z", StepKind.COMPARE, ()),
    )
    dag = PlanDAG(1, "plan-order", steps)
    assert dag.topological_order == ("b", "a", "z")
    assert dag.steps is steps
    with pytest.raises(ValueError, match="canonical topological"):
        replace(dag, steps=(steps[0], steps[2], steps[1]))
    with pytest.raises(ValueError, match="canonical topological"):
        replace(plan(), steps=tuple(reversed(plan().steps)))


def test_complete_dag_accepts_local_capacity_and_rejects_oversize() -> None:
    steps = tuple(
        PlanStep(1, str(index), StepKind.CHECK, tuple(str(i) for i in range(index)))
        for index in range(MAX_STEPS)
    )
    maximum = PlanDAG(1, "maximum", steps)
    assert sum(len(item.depends_on) for item in steps) == MAX_EDGES
    assert PlanDAG.from_json(maximum.canonical_bytes()) == maximum
    assert len(decision().canonical_bytes()) < MAX_JSON_BYTES
    for incomplete in ((), steps[:1]):
        with pytest.raises(ValueError, match="at least two"):
            PlanDAG(1, "incomplete", incomplete)
    with pytest.raises(ValueError, match="bound"):
        PlanDAG(1, "too-many", steps + (PlanStep(1, "extra", StepKind.CHECK, ()),))
    too_many_edges = (
        PlanStep(1, "a", StepKind.CHECK, ("b", "c", "d")),
        PlanStep(1, "b", StepKind.CHECK, ("a", "c", "d")),
        PlanStep(1, "c", StepKind.CHECK, ("a",)),
        PlanStep(1, "d", StepKind.CHECK, ()),
    )
    with pytest.raises(ValueError, match="edges"):
        PlanDAG(1, "too-many-edges", too_many_edges)


@pytest.mark.parametrize("size", (2, 3, 4))
def test_cycles_are_rejected_at_each_local_legal_node_count(size: int) -> None:
    cyclic = tuple(
        PlanStep(1, str(i), StepKind.CHECK, (str((i + 1) % size),)) for i in range(size)
    )
    with pytest.raises(ValueError, match="cycle"):
        PlanDAG(1, "cycle", cyclic)


def test_dependency_sets_reject_self_duplicate_unsorted_and_dangling_ids() -> None:
    for deps in (("step",), ("a", "a"), ("z", "a"), ("a", "b", "c", "d")):
        with pytest.raises(ValueError):
            PlanStep(1, "step", StepKind.ASSESS, deps)
    with pytest.raises(ValueError, match="dangling"):
        PlanDAG(
            1,
            "dangling",
            (
                PlanStep(1, "a", StepKind.CHECK, ()),
                PlanStep(1, "b", StepKind.CHECK, ("absent",)),
            ),
        )
    step = PlanStep(1, "a", StepKind.CHECK, ())
    for duplicate in (step, replace(step, kind=StepKind.COMPARE)):
        with pytest.raises(ValueError, match="duplicate"):
            PlanDAG(1, "duplicate", (step, duplicate))


def test_candidate_shapes_and_decision_cardinality_are_closed() -> None:
    record = decision()
    a, b, noop = record.candidates
    for changed in (
        (),
        (a, b),
        (a, b, noop, replace(noop, candidate_id="extra")),
        (a, b, replace(b, candidate_id="candidate-c")),
        (a, noop, replace(noop, candidate_id="noop-2")),
        (noop, b, a),
        (a, replace(b, candidate_id=a.candidate_id), noop),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(record, candidates=changed)
    with pytest.raises(ValueError):
        replace(noop, plan=plan())
    with pytest.raises(TypeError):
        replace(a, plan=None)
    with pytest.raises(TypeError):
        replace(a, goal=a.value)
    with pytest.raises(TypeError):
        replace(a, kind="plan")
    with pytest.raises(TypeError):
        replace(plan().steps[0], kind="check")


def test_duplicate_plan_id_and_content_aliases_cannot_fake_competition() -> None:
    record = decision()
    a, b, noop = record.candidates
    for aliased in (
        replace(plan(1), plan_id=plan().plan_id),
        replace(plan(), plan_id="different-id-same-steps"),
    ):
        with pytest.raises(ValueError, match="distinct|identical"):
            replace(record, candidates=(a, replace(b, plan=aliased), noop))


def test_selection_rejects_absent_id_wrong_digest_and_stale_candidate_content() -> None:
    record = decision()
    a, b, noop = record.candidates
    for selected in (
        replace(record.selected, candidate_id="missing"),
        replace(record.selected, candidate_id=b.candidate_id),
        replace(record.selected, candidate_digest="0" * 64),
    ):
        with pytest.raises(ValueError, match="absent|stale"):
            replace(record, selected=selected)
    updated_plan = replace(
        plan(),
        steps=(plan().steps[0], replace(plan().steps[1], kind=StepKind.ASSESS)),
    )
    updated_a = replace(a, plan=updated_plan)
    with pytest.raises(ValueError, match="stale"):
        replace(record, candidates=(updated_a, b, noop))
    fresh = replace(
        record,
        candidates=(updated_a, b, noop),
        selected=CandidateSelection(1, a.candidate_id, updated_a.content_digest),
    )
    assert fresh.selected.candidate_digest != record.selected.candidate_digest


def test_internal_goal_value_revision_digest_or_id_conflicts_are_rejected() -> None:
    record = decision()
    a, b, noop = record.candidates
    goal = b.goal
    value = b.value
    conflicting = (
        replace(b, goal=replace(goal, revision=goal.revision + 1)),
        replace(b, goal=replace(goal, goal_id="d" * 64)),
        replace(b, goal=replace(goal, goal_digest="d" * 64)),
        replace(b, value=replace(value, revision=value.revision + 1)),
        replace(b, value=replace(value, value_id="different-value")),
        replace(b, value=replace(value, value_digest="d" * 64)),
    )
    for conflict in conflicting:
        with pytest.raises(ValueError, match="snapshot"):
            replace(record, candidates=(a, conflict, noop))


def test_hold_is_not_a_selected_decision_or_no_op() -> None:
    record = decision()
    hold = HoldAssessment(1, ReasonCode.CONFLICTING_EVIDENCE, UNCERTAINTY_SCALE)
    assert hold.explanation().status is ExplanationStatus.HOLD
    assert hold.explanation().category is ExplanationCategory.REFUSAL
    assert not hasattr(hold, "selected")
    for refusal in (ReasonCode.CONFLICTING_EVIDENCE, ReasonCode.INSUFFICIENT_EVIDENCE):
        with pytest.raises(ValueError, match="not HOLD"):
            replace(record, reason=refusal)
    with pytest.raises(ValueError):
        replace(record, reason=ReasonCode.NO_OP_PREFERRED)
    with pytest.raises(TypeError):
        replace(record, selected=hold)
    with pytest.raises(TypeError):
        replace(record, candidates=(record.candidates[0], record.candidates[1], hold))
    with pytest.raises(ValueError):
        HoldAssessment(1, ReasonCode.PLAN_PREFERRED, 0)
    with pytest.raises(ValueError):
        DecisionRecord.from_value(hold.canonical_value())
    raw = record.canonical_value()
    raw["selected"] = "HOLD"
    with pytest.raises((TypeError, ValueError)):
        DecisionRecord.from_value(seal(raw, DECISION_DOMAIN))


def test_explanation_is_private_data_free_and_not_publisher_proof() -> None:
    record = decision()
    projection = record.explanation()
    raw = projection.canonical_value()
    assert set(raw) == {
        "schema_version",
        "category",
        "status",
        "reason",
        "uncertainty",
        "content_digest",
    }
    for secret in (
        GOAL_ID,
        GOAL_DIGEST,
        VALUE_ID,
        VALUE_DIGEST,
        record.decision_id,
        record.candidates[0].candidate_id,
        record.candidates[0].content_digest,
        plan().plan_id,
        plan().content_digest,
    ):
        assert secret.encode() not in projection.canonical_bytes()
    # Different private identities/digests cannot change the public projection.
    options = tuple(
        replace(
            item,
            goal=GoalEvidenceRef(1, "e" * 64, 21, "f" * 64),
            value=ValueEvidenceRef(1, "another-private-value", 22, "0" * 64),
        )
        for item in record.candidates
    )
    different = replace(
        record,
        decision_id="another-private-decision",
        candidates=options,
        selected=CandidateSelection(
            1, options[0].candidate_id, options[0].content_digest
        ),
    )
    assert different.explanation().canonical_bytes() == projection.canonical_bytes()
    assert ExplanationProjection.from_value(raw) == projection
    private_fields = (
        "text",
        "goal_id",
        "value_id",
        "source_digest",
        "publisher_authority",
    )
    for field in private_fields:
        with pytest.raises(ValueError):
            forged = seal(dict(raw, **{field: "private"}), EXPLANATION_DOMAIN)
            ExplanationProjection.from_value(forged)
    for changes in (
        {"category": ExplanationCategory.REFUSAL},
        {"status": ExplanationStatus.HOLD},
        {"reason": ReasonCode.NO_OP_PREFERRED},
        {"reason": "private free text"},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(projection, **changes)


@pytest.mark.parametrize("bad", (0, 2, True, 1.0, "1", None))
def test_inexact_or_future_versions_fail_with_recomputed_digest(bad: object) -> None:
    for value, domain in (
        (plan(), PLAN_DOMAIN),
        (decision(), DECISION_DOMAIN),
        (decision().explanation(), EXPLANATION_DOMAIN),
    ):
        with pytest.raises(ValueError):
            replace(value, schema_version=bad)
        raw = value.canonical_value()
        raw["schema_version"] = bad
        with pytest.raises(ValueError):
            type(value).from_value(seal(raw, domain))


def test_nested_future_version_cannot_be_hidden_by_resealing_parent() -> None:
    raw = plan().canonical_value()
    steps = cast(list[dict[str, object]], raw["steps"])
    steps[0]["schema_version"] = 2
    steps[0] = seal(steps[0], b"PROJECT-SUZKA:R16:PLAN-STEP:V1\0")
    with pytest.raises(ValueError):
        PlanDAG.from_value(seal(raw, PLAN_DOMAIN))


def test_closed_shape_rejects_unknown_missing_fields_and_invalid_enum() -> None:
    record = decision()
    raw = record.canonical_value()
    missing_reason = {key: item for key, item in raw.items() if key != "reason"}
    for bad in (dict(raw, hold=True), missing_reason):
        with pytest.raises(ValueError):
            DecisionRecord.from_value(seal(bad, DECISION_DOMAIN))
    raw["reason"] = "execute_tool"
    with pytest.raises(ValueError):
        DecisionRecord.from_value(seal(raw, DECISION_DOMAIN))
    dag = plan().canonical_value()
    steps = cast(list[dict[str, object]], dag["steps"])
    steps[0]["command"] = "private executable text"
    steps[0] = seal(steps[0], b"PROJECT-SUZKA:R16:PLAN-STEP:V1\0")
    with pytest.raises(ValueError):
        PlanDAG.from_value(seal(dag, PLAN_DOMAIN))


def test_json_rejects_noncanonical_or_malformed_encodings_and_collisions() -> None:
    good = decision().canonical_bytes()
    value = decision().canonical_value()
    reverse_keys = dict(reversed(tuple(value.items())))
    malformed = (
        good + b" ",
        b"\xef\xbb\xbf" + good,
        b"\xff",
        good[:-1],
        json.dumps(value, indent=2).encode(),
        json.dumps(reverse_keys, separators=(",", ":")).encode(),
        good.replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1
        ),
        good.replace(b'"uncertainty":250000', b'"uncertainty":NaN'),
        good.replace(b'"uncertainty":250000', b'"uncertainty":Infinity'),
        good.replace(b'"uncertainty":250000', b'"uncertainty":-Infinity'),
        good.replace(b'"uncertainty":250000', b'"uncertainty":1e999'),
        good.replace(b"private-decision-1", b"private-\\u0064ecision-1"),
        b"[" * 40 + b"0" + b"]" * 40,
        b" " * (MAX_JSON_BYTES + 1),
    )
    for payload in malformed:
        with pytest.raises((TypeError, ValueError)):
            DecisionRecord.from_json(payload)
    with pytest.raises(ValueError):
        DecisionRecord.from_json(bytearray(good))
    with pytest.raises(ValueError):
        DecisionRecord.from_json(good.decode())


@pytest.mark.parametrize("bad", (True, False, -1, 1.0, 1_000_001, "0", None))
def test_uncertainty_is_an_exact_bounded_integer(bad: object) -> None:
    for value in (
        decision(),
        decision().explanation(),
        HoldAssessment(1, ReasonCode.INSUFFICIENT_EVIDENCE, 0),
    ):
        with pytest.raises(ValueError):
            replace(value, uncertainty=bad)
    raw = decision().canonical_value()
    raw["uncertainty"] = bad
    with pytest.raises(ValueError):
        DecisionRecord.from_value(seal(raw, DECISION_DOMAIN))


def test_uncertainty_endpoints_and_evidence_revision_limits_are_representable() -> None:
    for endpoint in (0, UNCERTAINTY_SCALE):
        assert replace(decision(), uncertainty=endpoint).uncertainty == endpoint
    for revision in (0, 2**31 - 1):
        goal = GoalEvidenceRef(1, GOAL_ID, revision, GOAL_DIGEST)
        value = ValueEvidenceRef(1, VALUE_ID, revision, VALUE_DIGEST)
        assert GoalEvidenceRef.from_json(goal.canonical_bytes()) == goal
        assert ValueEvidenceRef.from_json(value.canonical_bytes()) == value
    for revision in (True, -1, 2**31, 1.0):
        with pytest.raises(ValueError):
            GoalEvidenceRef(1, GOAL_ID, revision, GOAL_DIGEST)
        with pytest.raises(ValueError):
            ValueEvidenceRef(1, VALUE_ID, revision, VALUE_DIGEST)


def test_ids_and_evidence_digests_are_exact_bounded_opaque_values() -> None:
    class StringSubclass(str):
        pass

    bad_ids = (
        "",
        " spaced",
        "path/id",
        "a..b",
        "x" * 129,
        "日本語",
        StringSubclass("a"),
    )
    for bad in bad_ids:
        with pytest.raises(ValueError):
            replace(plan(), plan_id=bad)
        with pytest.raises(ValueError):
            ValueEvidenceRef(1, bad, 0, VALUE_DIGEST)
    assert replace(plan(), plan_id="x" * 128).plan_id == "x" * 128
    for bad_digest in ("A" * 64, "g" * 64, "a" * 63, "a" * 65, True):
        with pytest.raises(ValueError):
            GoalEvidenceRef(1, bad_digest, 0, GOAL_DIGEST)
        with pytest.raises(ValueError):
            ValueEvidenceRef(1, VALUE_ID, 0, bad_digest)
        with pytest.raises(ValueError):
            CandidateSelection(1, "candidate-a", bad_digest)


def test_exports_and_direct_imports_remain_pure_without_runtime_authority() -> None:
    tree = ast.parse(inspect.getsource(contracts))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imported == {
        "__future__",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "typing",
        "suzka.identifiers",
        "suzka.limits",
    }
    assert "suzka.r15.common" not in imported
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
    assert set(r16.__all__) == {
        "CandidateKind",
        "CandidateSelection",
        "DecisionRecord",
        "ExplanationCategory",
        "ExplanationProjection",
        "ExplanationStatus",
        "GoalEvidenceRef",
        "HoldAssessment",
        "PlanCandidate",
        "PlanDAG",
        "PlanStep",
        "ReasonCode",
        "StepKind",
        "ValueEvidenceRef",
    }
    for name in (
        "GoalSubjectAdmission",
        "CommitmentSubjectAdmission",
        "StateWAL",
        "MainLoop",
        "ActionIntent",
        "Outcome",
    ):
        assert not hasattr(contracts, name)
    for value in (decision(), HoldAssessment(1, ReasonCode.INSUFFICIENT_EVIDENCE, 0)):
        for name in ("commit", "execute", "admit", "publish", "validate_publisher"):
            assert not hasattr(value, name)
