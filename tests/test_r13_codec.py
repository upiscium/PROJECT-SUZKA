"""Lossless R13 persistence codec and Gate 0 capacity checks."""

import ast
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import suzka.runtime.r13_codec as r13_codec_module
from suzka.motivation.commitment import (
    CommitmentLifecycle,
    CommitmentRevisionOperation,
)
from suzka.motivation.commitment_system import CommitmentSystem
from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.goal import (
    GoalLifecycle,
    GoalRevisionOperation,
    GoalRevisionReason,
)
from suzka.motivation.goal_system import GoalSystem
from suzka.motivation.motivation import MotivationKind
from suzka.motivation.system import MotivationSystem, _event_input_digest
from suzka.runtime.agent_state import (
    AGENT_STATE_FUTURE_STATE_RESERVE_BYTES,
    AGENT_STATE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES,
    project_agent_state_schema_max_bytes,
)
from suzka.runtime.r13_codec import (
    R13GraphSnapshot,
    R13StateSnapshot,
    r13_codec_schema_maxima,
)
from test_commitment_system import (
    BASE_TIME as COMMITMENT_BASE_TIME,
    admission_for as commitment_admission_for,
    ingest as ingest_commitment,
    make_proposal as make_commitment_proposal,
    transition_for as commitment_transition_for,
)
from test_goal_system import (
    admission_for as goal_admission_for,
    ingest as ingest_goal,
    make_proposal as make_goal_proposal,
    transition_for as goal_transition_for,
)
from test_motivation_system import (
    make_candidate as make_motivation_candidate,
    make_evidence as make_motivation_evidence,
    source_event as motivation_source_event,
)


def _motivation_snapshot():
    system = MotivationSystem()
    target = R13Reference(R13ReferenceKind.VALUE, "value:codec-shared")
    experience = make_motivation_evidence(
        R13Reference(R13ReferenceKind.EXPERIENCE, "source:codec-shared"),
        target=target,
    )
    first_event = motivation_source_event(experience, 1)
    candidate = make_motivation_candidate(experience, first_event)
    system.apply_evidence(experience, first_event, candidate=candidate)

    belief = make_motivation_evidence(
        R13Reference(R13ReferenceKind.BELIEF, "source:codec-shared"),
        target=target,
        kind=MotivationKind.DESIRE,
    )
    system.apply_evidence(belief, motivation_source_event(belief, 2))
    return system.snapshot()


def _goal_snapshot():
    system = GoalSystem()
    proposal = make_goal_proposal(description="make steady progress 🧭")
    ingest_goal(system, proposal, 1)
    admission, admission_event = goal_admission_for(proposal, 2)
    record = system.adopt(proposal.goal_id, admission, admission_event)

    for sequence in range(3, 13):
        if record.lifecycle is GoalLifecycle.ADOPTED:
            proof, event = goal_transition_for(
                record,
                sequence,
                GoalRevisionOperation.DEFER,
                GoalRevisionReason.SUBJECT_DEFERRED,
            )
            record = system.defer(record.goal_id, proof, event)
        else:
            proof, event = goal_transition_for(
                record,
                sequence,
                GoalRevisionOperation.ADOPT,
                GoalRevisionReason.SUBJECT_ADMISSION,
            )
            record = system.resume(record.goal_id, proof, event)
    return system.snapshot()


def _commitment_snapshot():
    system = CommitmentSystem()

    release_proposal = make_commitment_proposal("codec-release")
    ingest_commitment(system, release_proposal, 1)
    release_admission, release_admission_event = commitment_admission_for(
        release_proposal, 2
    )
    release_active = system.accept(
        release_proposal.commitment_id,
        release_admission,
        release_admission_event,
    )

    renegotiate_proposal = make_commitment_proposal(
        "codec-renegotiate",
        genesis_sequence=3,
        genesis_at=COMMITMENT_BASE_TIME + timedelta(seconds=3),
    )
    ingest_commitment(system, renegotiate_proposal, 3)
    renegotiate_admission, renegotiate_admission_event = commitment_admission_for(
        renegotiate_proposal, 4
    )
    renegotiate_active = system.accept(
        renegotiate_proposal.commitment_id,
        renegotiate_admission,
        renegotiate_admission_event,
    )

    release_proof, release_event = commitment_transition_for(
        release_active, 5, CommitmentRevisionOperation.RELEASE
    )
    system.release(release_active.commitment_id, release_proof, release_event)
    renegotiate_proof, renegotiate_event = commitment_transition_for(
        renegotiate_active, 6, CommitmentRevisionOperation.RENEGOTIATE
    )
    system.renegotiate(
        renegotiate_active.commitment_id,
        renegotiate_proof,
        renegotiate_event,
    )
    return system.snapshot()


@pytest.fixture(scope="module")
def domain_snapshots():
    return {
        "motivation": _motivation_snapshot(),
        "goal": _goal_snapshot(),
        "commitment": _commitment_snapshot(),
    }


@pytest.mark.parametrize("domain", ("motivation", "goal", "commitment"))
def test_empty_and_populated_domain_snapshots_round_trip_exactly(
    domain: str, domain_snapshots
) -> None:
    empty = R13StateSnapshot.empty()
    empty_graph = getattr(empty, domain)
    assert empty_graph.domain == domain
    empty_value = empty_graph.restore()
    assert type(empty_value).__name__ == {
        "motivation": "MotivationSystemSnapshot",
        "goal": "GoalSystemSnapshot",
        "commitment": "CommitmentSystemSnapshot",
    }[domain]
    assert R13GraphSnapshot.capture(empty_value) == empty_graph

    snapshot = domain_snapshots[domain]
    graph = R13GraphSnapshot.capture(snapshot)
    restored = graph.restore()

    assert graph.domain == domain
    assert restored == snapshot
    assert restored.authority_digest == snapshot.authority_digest
    assert R13GraphSnapshot.capture(restored) == graph


def test_motivation_table_keeps_candidate_evidence_eligibility_and_typed_refs(
    domain_snapshots,
) -> None:
    snapshot = domain_snapshots["motivation"]
    graph = R13GraphSnapshot.capture(snapshot)
    node_types = {row[0] for row in graph.nodes}

    assert {
        "MotivationInterpretationCandidate",
        "MotivationEvidence",
        "MotivationGoalProposalWitness",
    } <= node_types
    assert len(snapshot.candidates) == 1
    assert len(snapshot.evidence_ledger) == 2
    assert len(snapshot.goal_proposal_witnesses) == 1
    witness = snapshot.goal_proposal_witnesses[0]
    assert witness.event_sequence == 1
    assert witness.event_id == "event:1"
    assert witness.evidence_ref == R13Reference(
        R13ReferenceKind.EXPERIENCE, "source:codec-shared"
    )

    typed_refs = {
        entry.evidence.source_ref for entry in snapshot.evidence_ledger
    }
    assert typed_refs == {
        R13Reference(R13ReferenceKind.EXPERIENCE, "source:codec-shared"),
        R13Reference(R13ReferenceKind.BELIEF, "source:codec-shared"),
    }
    shared_text_index = graph.strings.index("source:codec-shared")
    encoded_shared_refs = {
        (row[1][0], row[1][1])
        for row in graph.nodes
        if row[0] == "R13Reference" and row[1][1] == shared_text_index
    }
    assert encoded_shared_refs == {
        (R13ReferenceKind.EXPERIENCE.value, shared_text_index),
        (R13ReferenceKind.BELIEF.value, shared_text_index),
    }


def test_goal_and_commitment_admissions_and_proofs_are_retained_in_nodes(
    domain_snapshots,
) -> None:
    goal_snapshot = domain_snapshots["goal"]
    goal_graph = R13GraphSnapshot.capture(goal_snapshot)
    goal_record = goal_snapshot.records[0]
    assert goal_record.history_anchor is not None
    assert goal_record.subject_admission is not None
    assert goal_record.subject_transition_proofs
    assert {"GoalSubjectAdmission", "GoalSubjectTransitionProof"} <= {
        row[0] for row in goal_graph.nodes
    }
    assert goal_record.subject_admission.admission_digest in goal_graph.strings
    assert all(
        proof.transition_digest in goal_graph.strings
        for proof in goal_record.subject_transition_proofs
    )
    assert goal_record.description in goal_graph.strings

    commitment_snapshot = domain_snapshots["commitment"]
    commitment_graph = R13GraphSnapshot.capture(commitment_snapshot)
    assert {record.lifecycle for record in commitment_snapshot.records} == {
        CommitmentLifecycle.RELEASED,
        CommitmentLifecycle.RENEGOTIATED,
    }
    assert {"CommitmentSubjectAdmission", "CommitmentSubjectTransitionProof"} <= {
        row[0] for row in commitment_graph.nodes
    }
    for record in commitment_snapshot.records:
        assert record.subject_admission is not None
        assert record.subject_admission.admission_digest in commitment_graph.strings
        assert record.subject_transition_proofs
        assert all(
            proof.transition_digest in commitment_graph.strings
            for proof in record.subject_transition_proofs
        )
        assert all(
            scope_item in commitment_graph.strings for scope_item in record.scope
        )


def _assert_invalid_graph(graph: R13GraphSnapshot, mutate) -> None:
    payload = deepcopy(graph.model_dump(mode="python"))
    mutate(payload)
    with pytest.raises(ValidationError):
        R13GraphSnapshot.model_validate(payload)


def test_computed_digest_string_and_backward_index_tampering_are_rejected(
    domain_snapshots,
) -> None:
    graph = R13GraphSnapshot.capture(domain_snapshots["goal"])

    def tamper_digest(payload) -> None:
        payload["strings"] = list(payload["strings"])
        digest_index = payload["nodes"][-1][2][0]
        original = payload["strings"][digest_index]
        replacement = f"tampered:{original}"
        while replacement in payload["strings"]:
            replacement += "x"
        payload["strings"][digest_index] = replacement

    _assert_invalid_graph(graph, tamper_digest)

    def tamper_string(payload) -> None:
        payload["strings"] = list(payload["strings"])
        payload["strings"][0] += "-tampered"

    _assert_invalid_graph(graph, tamper_string)

    def tamper_index(payload) -> None:
        payload["nodes"] = [list(row) for row in payload["nodes"]]
        payload["nodes"][-1][1][0][0] = len(payload["nodes"]) - 1

    _assert_invalid_graph(graph, tamper_index)


def test_future_schema_and_node_types_are_rejected(domain_snapshots) -> None:
    graph = R13GraphSnapshot.capture(domain_snapshots["goal"])

    def future_schema(payload) -> None:
        payload["schema_version"] = 2

    _assert_invalid_graph(graph, future_schema)

    def future_node(payload) -> None:
        payload["nodes"] = [list(row) for row in payload["nodes"]]
        payload["nodes"].insert(-1, ["FutureR13Node", [], []])

    _assert_invalid_graph(graph, future_node)


def test_duplicate_unreachable_and_reordered_nodes_or_strings_are_rejected(
    domain_snapshots,
) -> None:
    graph = R13GraphSnapshot.capture(domain_snapshots["goal"])

    def duplicate_node(payload) -> None:
        payload["nodes"] = [list(row) for row in payload["nodes"]]
        payload["nodes"].insert(-1, deepcopy(payload["nodes"][0]))

    _assert_invalid_graph(graph, duplicate_node)

    def unreachable_node(payload) -> None:
        payload["strings"] = list(payload["strings"])
        payload["strings"].append("unreachable:reference")
        payload["nodes"] = [list(row) for row in payload["nodes"]]
        payload["nodes"].insert(
            -1,
            [
                "R13Reference",
                [R13ReferenceKind.EVENT.value, len(payload["strings"]) - 1],
                [],
            ],
        )

    _assert_invalid_graph(graph, unreachable_node)

    def reordered_node(payload) -> None:
        payload["nodes"] = [list(row) for row in payload["nodes"]]
        payload["nodes"][0], payload["nodes"][1] = (
            payload["nodes"][1],
            payload["nodes"][0],
        )

    _assert_invalid_graph(graph, reordered_node)

    def duplicate_string(payload) -> None:
        payload["strings"] = list(payload["strings"])
        payload["strings"].append(payload["strings"][0])

    _assert_invalid_graph(graph, duplicate_string)

    def reordered_string(payload) -> None:
        payload["strings"] = list(payload["strings"])
        payload["strings"][0], payload["strings"][1] = (
            payload["strings"][1],
            payload["strings"][0],
        )

    _assert_invalid_graph(graph, reordered_string)


def test_state_snapshot_requires_all_domains_and_forbids_private_extra_fields() -> None:
    state = R13StateSnapshot.empty()
    payload = state.model_dump(mode="python")

    missing_version = dict(payload)
    missing_version.pop("schema_version")
    with pytest.raises(ValidationError):
        R13StateSnapshot.model_validate(missing_version)
    graph_without_version = dict(payload["goal"])
    graph_without_version.pop("schema_version")
    with pytest.raises(ValidationError):
        R13GraphSnapshot.model_validate(graph_without_version)

    with pytest.raises(ValidationError):
        R13StateSnapshot.model_validate(
            {"motivation": payload["motivation"], "goal": payload["goal"]}
        )
    with pytest.raises(ValidationError):
        R13StateSnapshot.model_validate({**payload, "private_reasoning": "hidden"})
    with pytest.raises(ValidationError):
        R13GraphSnapshot.model_validate(
            {
                **payload["motivation"],
                "raw_prompt": "not an R13 persistence field",
            }
        )


def test_malformed_codec_resource_shapes_are_bounded_before_reconstruction() -> None:
    graph = R13StateSnapshot.empty().goal
    payload = graph.model_dump(mode="python")
    payload["strings"] = (*payload["strings"], "x" * 1025)
    with pytest.raises(ValidationError):
        R13GraphSnapshot.model_validate(payload)
    maximum = max(r13_codec_schema_maxima()[name] for name in ("motivation", "goal", "commitment"))
    with pytest.raises(ValueError, match="schema byte bound"):
        R13GraphSnapshot.model_validate_json(b" " * (maximum + 1))
    payload = graph.model_dump(mode="python")
    payload["nodes"][0][1][0] = [[[[]]]]
    with pytest.raises(ValidationError):
        R13GraphSnapshot.model_validate(payload)
    payload = graph.model_dump(mode="python")
    payload["nodes"][0][1][0] = [0] * 1025
    with pytest.raises(ValidationError):
        R13GraphSnapshot.model_validate(payload)


@pytest.mark.parametrize("mismatch", ["time", "same_sequence_id"])
def test_invalid_source_provenance_cannot_enter_persisted_authority(mismatch: str) -> None:
    original = _motivation_snapshot()
    first = original.evidence_ledger[0]
    receipt = next(item for item in original.event_receipts if item.event_id == first.event_id)
    evidence = replace(
        first.evidence,
        source_event_id="event:source-observation",
        source_event_sequence=1 if mismatch == "time" else first.event_sequence,
        observed_at=receipt.recorded_at + timedelta(days=3)
        if mismatch == "time" else receipt.recorded_at,
    )
    altered_receipt = replace(
        receipt,
        evidence_digest=evidence.evidence_digest,
        input_digest=_event_input_digest(
            receipt.operation,
            {"candidate_digest": receipt.candidate_digest,
             "evidence_digest": evidence.evidence_digest},
        ),
    )
    error = "source observation is in the future" if mismatch == "time" else "source event ID differs"
    with pytest.raises(ValueError, match=error):
        replace(
            original,
            evidence_ledger=tuple(
                replace(item, evidence=evidence) if item is first else item
                for item in original.evidence_ledger
            ),
            event_receipts=tuple(
                altered_receipt if item is receipt else item for item in original.event_receipts
            ),
        )


def test_capacity_gate_uses_derived_codec_maxima_and_utf8_domain_bytes(
    domain_snapshots,
) -> None:
    maxima = r13_codec_schema_maxima()
    assert maxima["motivation"] == 10_226_813
    assert maxima["goal"] == 10_133_579
    assert maxima["commitment"] == 3_386_967
    assert maxima["total"] == 23_747_415

    for domain, snapshot in domain_snapshots.items():
        graph = R13GraphSnapshot.capture(snapshot)
        encoded = graph.model_dump_json().encode("utf-8")
        assert len(encoded) <= maxima[domain]
        if domain == "goal":
            assert "🧭" in graph.model_dump_json()
            ascii_escaped = json.dumps(
                graph.model_dump(mode="json"),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            assert len(encoded) < len(ascii_escaped)

    assert maxima["total"] > sum(
        maxima[domain] for domain in ("motivation", "goal", "commitment")
    )
    assert AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES == 85_826_260
    assert AGENT_STATE_FUTURE_STATE_RESERVE_BYTES == 16 * 1024 * 1024
    assert AGENT_STATE_MAX_SERIALIZED_BYTES == 128 * 1024 * 1024
    projected = project_agent_state_schema_max_bytes(
        schema_version=8,
        base_schema_version=7,
        added_field_maxima={"r13_state": maxima["total"]},
    )
    assert projected == 126_350_904
    assert projected <= AGENT_STATE_MAX_SERIALIZED_BYTES
    assert (
        AGENT_STATE_MAX_SERIALIZED_BYTES
        - (projected - AGENT_STATE_FUTURE_STATE_RESERVE_BYTES)
        >= AGENT_STATE_FUTURE_STATE_RESERVE_BYTES
    )


def test_codec_has_no_model_memory_or_policy_runtime_dependencies() -> None:
    source_path = Path(r13_codec_module.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)

    forbidden_prefixes = (
        "suzka.models",
        "suzka.memory",
        "suzka.policy",
        "suzka.scheduler",
        "suzka.runtime.agent_state",
        "torch",
        "transformers",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in forbidden_prefixes
    )
    assert not imported_names.intersection(
        {
            "MotivationSystem",
            "GoalSystem",
            "CommitmentSystem",
            "MemorySystem",
            "AgentStateStore",
        }
    )
