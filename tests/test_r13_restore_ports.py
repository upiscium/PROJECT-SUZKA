"""Exact U5 restore-port coverage for R13 Motivation, Goal, and Commitment."""

from dataclasses import replace

import pytest

from suzka.motivation.commitment import CommitmentLifecycle
from suzka.motivation.commitment_system import (
    CommitmentMutationEvidence,
    CommitmentSystem,
    CommitmentSystemEventOperation,
    CommitmentSystemSnapshot,
)
from suzka.motivation.goal import (
    GoalLifecycle,
    GoalSubjectTransitionProof,
)
from suzka.motivation.goal_system import (
    GoalMutationEvidence,
    GoalSystem,
    GoalSystemEventOperation,
    GoalSystemSnapshot,
)
from suzka.motivation.system import (
    MotivationEventOperation,
    MotivationMutationEvidence,
    MotivationSystem,
    MotivationSystemSnapshot,
)
from suzka.runtime.r13_codec import R13GraphSnapshot
from test_r13_codec import (
    _commitment_snapshot,
    _goal_snapshot,
    _motivation_snapshot,
)


def _restore_graph(snapshot: object) -> object:
    graph = R13GraphSnapshot.capture(snapshot)
    return graph.restore()


def _goal_event(receipt) -> GoalMutationEvidence:
    return GoalMutationEvidence(
        event_id=receipt.event_id,
        event_sequence=receipt.event_sequence,
        recorded_at=receipt.recorded_at,
        evidence_refs=receipt.evidence_refs,
    )


def _commitment_event(receipt) -> CommitmentMutationEvidence:
    return CommitmentMutationEvidence(
        event_id=receipt.event_id,
        event_sequence=receipt.event_sequence,
        recorded_at=receipt.recorded_at,
        evidence_refs=receipt.evidence_refs,
    )


def _assert_restore_rejections_are_atomic(
    *,
    snapshot: object,
    wrong_type_snapshot: object,
    export,
    restore,
    malformed_field: str,
) -> None:
    before = export()

    malformed = replace(snapshot)
    object.__setattr__(malformed, malformed_field, ())
    with pytest.raises((TypeError, ValueError)):
        restore(malformed)
    assert export() == before

    with pytest.raises(TypeError):
        restore(wrong_type_snapshot)
    assert export() == before

    future = replace(snapshot)
    object.__setattr__(future, "schema_version", snapshot.schema_version + 1)
    with pytest.raises((TypeError, ValueError)):
        restore(future)
    assert export() == before


def test_motivation_restore_preserves_exact_graph_and_event_replay() -> None:
    original = _motivation_snapshot()
    restored = _restore_graph(original)
    assert type(restored) is MotivationSystemSnapshot
    assert restored == original
    assert restored.authority_digest == original.authority_digest

    system = MotivationSystem()
    system.restore_motivation_state(restored)
    assert system.export_motivation_state() == original
    before_replay = system.export_motivation_state()

    receipt = next(
        item
        for item in original.event_receipts
        if item.operation is MotivationEventOperation.EVIDENCE
        and item.candidate_digest is not None
    )
    assert receipt.goal_proposal_digest is not None
    assert any(
        item.event_id == receipt.event_id
        and item.proposal_digest == receipt.goal_proposal_digest
        for item in original.goal_proposal_witnesses
    )
    ledger_entry = next(
        item
        for item in original.evidence_ledger
        if item.event_id == receipt.event_id
        and item.event_sequence == receipt.event_sequence
    )
    candidate = next(
        item
        for item in original.candidates
        if item.candidate_digest == receipt.candidate_digest
    )
    event = MotivationMutationEvidence(
        event_id=receipt.event_id,
        event_sequence=receipt.event_sequence,
        recorded_at=receipt.recorded_at,
        evidence_refs=receipt.evidence_refs,
    )

    system.apply_evidence(ledger_entry.evidence, event, candidate=candidate)
    system.ingest_candidate(candidate, event)
    assert system.export_motivation_state() == before_replay

    _assert_restore_rejections_are_atomic(
        snapshot=original,
        wrong_type_snapshot=GoalSystemSnapshot(),
        export=system.export_motivation_state,
        restore=system.restore_motivation_state,
        malformed_field="evidence_ledger",
    )

    empty = MotivationSystemSnapshot((), (), (), ())
    system.restore_motivation_state(empty)
    assert system.export_motivation_state() == empty
    assert system.records == ()


def test_goal_restore_preserves_ingestion_admission_replay_and_compaction() -> None:
    original = _goal_snapshot()
    restored = _restore_graph(original)
    assert type(restored) is GoalSystemSnapshot
    assert restored == original
    assert restored.authority_digest == original.authority_digest
    assert any(record.history_anchor is not None for record in original.records)

    system = GoalSystem()
    system.restore_goal_state(restored)
    assert system.export_goal_state() == original
    before_replay = system.export_goal_state()

    ingestion = next(
        item
        for item in original.event_receipts
        if item.operation is GoalSystemEventOperation.INGEST_PROPOSAL
    )
    current = system.get(ingestion.goal_id)
    assert current is not None
    assert ingestion.proposal_genesis is not None
    proposal = replace(
        current,
        lifecycle=GoalLifecycle.PROPOSED,
        outcome_evidence_refs=(),
        subject_admission=None,
        subject_transition_proofs=(),
        revision=0,
        revision_history=(ingestion.proposal_genesis,),
        history_anchor=None,
    )
    system.ingest_proposal(proposal, _goal_event(ingestion))

    admission_receipt = next(
        item
        for item in original.event_receipts
        if item.operation is GoalSystemEventOperation.ADOPT
    )
    admitted = system.get(admission_receipt.goal_id)
    assert admitted is not None
    assert admitted.subject_admission is not None
    system.adopt(
        admission_receipt.goal_id,
        admitted.subject_admission,
        _goal_event(admission_receipt),
    )

    transition_receipt = max(
        (
            item
            for item in original.event_receipts
            if item.operation
            in {
                GoalSystemEventOperation.DEFER,
                GoalSystemEventOperation.RESUME,
                GoalSystemEventOperation.ABANDON,
            }
        ),
        key=lambda item: item.event_sequence,
    )
    revision = transition_receipt.revision_witness
    assert revision is not None
    assert revision.previous_lifecycle_state is not None
    proof = GoalSubjectTransitionProof(
        goal_id=revision.goal_id,
        proposal_digest=revision.proposal_digest,
        operation=revision.operation,
        reason=revision.reason,
        previous_lifecycle_state=revision.previous_lifecycle_state,
        evidence_refs=tuple(item.reference for item in transition_receipt.evidence_refs),
        event_id=transition_receipt.event_id,
        event_sequence=transition_receipt.event_sequence,
    )
    assert proof.transition_digest == transition_receipt.transition_digest
    transition_methods = {
        GoalSystemEventOperation.DEFER: system.defer,
        GoalSystemEventOperation.RESUME: system.resume,
        GoalSystemEventOperation.ABANDON: system.abandon,
    }
    transition_methods[transition_receipt.operation](
        transition_receipt.goal_id,
        proof,
        _goal_event(transition_receipt),
    )
    assert system.export_goal_state() == before_replay

    _assert_restore_rejections_are_atomic(
        snapshot=original,
        wrong_type_snapshot=CommitmentSystemSnapshot(),
        export=system.export_goal_state,
        restore=system.restore_goal_state,
        malformed_field="records",
    )

    empty = GoalSystemSnapshot()
    system.restore_goal_state(empty)
    assert system.export_goal_state() == empty
    assert system.records == ()


def test_commitment_restore_preserves_proposal_and_terminal_proof_replay() -> None:
    original = _commitment_snapshot()
    restored = _restore_graph(original)
    assert type(restored) is CommitmentSystemSnapshot
    assert restored == original
    assert restored.authority_digest == original.authority_digest
    assert {item.lifecycle for item in original.records} == {
        CommitmentLifecycle.RELEASED,
        CommitmentLifecycle.RENEGOTIATED,
    }

    system = CommitmentSystem()
    system.restore_commitment_state(restored)
    assert system.export_commitment_state() == original
    before_replay = system.export_commitment_state()

    for record in original.records:
        ingestion = next(
            item
            for item in original.event_receipts
            if item.commitment_id == record.commitment_id
            and item.operation is CommitmentSystemEventOperation.INGEST_PROPOSAL
        )
        assert ingestion.proposal_genesis is not None
        proposal = replace(
            record,
            lifecycle=CommitmentLifecycle.PROPOSED,
            subject_admission=None,
            subject_transition_proofs=(),
            outcome_evidence_refs=(),
            revision=0,
            revision_history=(ingestion.proposal_genesis,),
            history_anchor=None,
        )
        system.ingest_proposal(proposal, _commitment_event(ingestion))

    for receipt in original.event_receipts:
        if receipt.operation is CommitmentSystemEventOperation.ACCEPT:
            assert receipt.admission is not None
            system.accept(
                receipt.commitment_id,
                receipt.admission,
                _commitment_event(receipt),
            )
        elif receipt.operation is CommitmentSystemEventOperation.RELEASE:
            assert receipt.transition_proof is not None
            system.release(
                receipt.commitment_id,
                receipt.transition_proof,
                _commitment_event(receipt),
            )
        elif receipt.operation is CommitmentSystemEventOperation.RENEGOTIATE:
            assert receipt.transition_proof is not None
            system.renegotiate(
                receipt.commitment_id,
                receipt.transition_proof,
                _commitment_event(receipt),
            )
    assert system.export_commitment_state() == before_replay

    _assert_restore_rejections_are_atomic(
        snapshot=original,
        wrong_type_snapshot=GoalSystemSnapshot(),
        export=system.export_commitment_state,
        restore=system.restore_commitment_state,
        malformed_field="records",
    )

    empty = CommitmentSystemSnapshot()
    system.restore_commitment_state(empty)
    assert system.export_commitment_state() == empty
    assert system.records == ()
