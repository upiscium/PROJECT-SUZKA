"""Focused tests for the process-local R13 GoalSystem authority."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import ast
from pathlib import Path
import sys

import pytest

import suzka.motivation.goal_system as goal_system_module
from suzka.motivation.common import (
    R13_MAX_EVENT_SEQUENCE,
    R13Reference,
    R13ReferenceKind,
)
from suzka.motivation.goal import (
    GoalAdmissionReason,
    GoalLifecycle,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    GoalSubjectAdmission,
    GoalSubjectTransitionProof,
    goal_id_for_target,
    goal_proposal_digest,
)
from suzka.motivation.goal_system import (
    GOAL_SYSTEM_MAX_EVENT_RECEIPTS,
    GOAL_SYSTEM_MAX_RECORDS,
    GOAL_SYSTEM_MAX_SERIALIZED_BYTES,
    GoalMutationEvidence,
    GoalSystem,
    GoalSystemCapacityExceeded,
    GoalSystemConflict,
    GoalSystemError,
    GoalSystemEventOperation,
    GoalSystemSnapshot,
)


BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def make_proposal(
    *,
    key: str = "health",
    description: str = "exercise regularly",
    genesis_event_id: str = "event:goal:genesis",
    genesis_sequence: int = 1,
    created_at: datetime = BASE_TIME,
    origin_kind: R13ReferenceKind = R13ReferenceKind.USER_REQUEST,
    origin_ref: str = "request:exercise",
    evidence_refs: tuple[R13Reference, ...] | None = None,
    dependencies: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> GoalRecord:
    target = R13Reference(R13ReferenceKind.VALUE, f"value:{key}")
    origins = (R13Reference(origin_kind, origin_ref),)
    evidence = evidence_refs or (
        R13Reference(R13ReferenceKind.EVENT, f"event:evidence:{key}"),
    )
    proposal_digest = goal_proposal_digest(
        target,
        description,
        origin_refs=origins,
        evidence_refs=evidence,
        dependencies=dependencies,
        conflicts=conflicts,
    )
    goal_id = goal_id_for_target(target)
    genesis = GoalRevisionRecord(
        goal_id=goal_id,
        revision=0,
        operation=GoalRevisionOperation.CREATE,
        reason=GoalRevisionReason.CREATION,
        created_at=created_at,
        previous_lifecycle_state=None,
        proposal_digest=proposal_digest,
        event_id=genesis_event_id,
        event_sequence=genesis_sequence,
        evidence_refs=tuple(sorted(item.reference for item in evidence)),
    )
    return GoalRecord(
        goal_id=goal_id,
        target=target,
        description=description,
        origin_refs=origins,
        evidence_refs=evidence,
        dependencies=dependencies,
        conflicts=conflicts,
        revision_history=(genesis,),
    )


def make_event(
    sequence: int,
    evidence_refs: tuple[R13Reference, ...],
    *,
    event_id: str | None = None,
    recorded_at: datetime | None = None,
) -> GoalMutationEvidence:
    return GoalMutationEvidence(
        event_id=event_id or f"event:mutation:{sequence}",
        event_sequence=sequence,
        recorded_at=recorded_at or BASE_TIME + timedelta(seconds=sequence),
        evidence_refs=evidence_refs,
    )


def ingest(
    system: GoalSystem,
    proposal: GoalRecord,
    sequence: int,
    *,
    event_id: str | None = None,
    recorded_at: datetime | None = None,
) -> GoalRecord:
    if sequence == proposal.revision_history[0].event_sequence:
        genesis = proposal.revision_history[0]
        if event_id is not None and event_id != genesis.event_id:
            raise AssertionError("same-sequence ingestion must use the exact genesis event")
        if recorded_at is not None and recorded_at != genesis.created_at:
            raise AssertionError("same-sequence ingestion must use the exact genesis time")
        event_id = genesis.event_id
        recorded_at = genesis.created_at
    return system.ingest_proposal(
        proposal,
        make_event(
            sequence,
            proposal.evidence_refs,
            event_id=event_id,
            recorded_at=recorded_at,
        ),
    )


def admission_for(
    proposal: GoalRecord,
    sequence: int,
    *,
    event_id: str | None = None,
) -> tuple[GoalSubjectAdmission, GoalMutationEvidence]:
    event = make_event(
        sequence,
        proposal.evidence_refs,
        event_id=event_id or f"event:admission:{sequence}",
    )
    admission = GoalSubjectAdmission(
        goal_id=proposal.goal_id,
        proposal_digest=proposal.proposal_digest,
        evidence_refs=tuple(sorted(item.reference for item in proposal.evidence_refs)),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        reason=GoalAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    return admission, event


def transition_for(
    record: GoalRecord,
    sequence: int,
    operation: GoalRevisionOperation,
    reason: GoalRevisionReason,
) -> tuple[GoalSubjectTransitionProof, GoalMutationEvidence]:
    event_refs = (R13Reference(R13ReferenceKind.EVENT, f"event:decision:{sequence}"),)
    event = make_event(sequence, event_refs)
    proof = GoalSubjectTransitionProof(
        goal_id=record.goal_id,
        proposal_digest=record.proposal_digest,
        operation=operation,
        reason=reason,
        previous_lifecycle_state=record.lifecycle,
        evidence_refs=tuple(sorted(item.reference for item in event_refs)),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
    )
    return proof, event


def adopted_system(
    *, key: str = "health", origin_kind: R13ReferenceKind = R13ReferenceKind.USER_REQUEST
) -> tuple[GoalSystem, GoalRecord]:
    system = GoalSystem()
    proposal = make_proposal(key=key, origin_kind=origin_kind)
    ingest(system, proposal, 1)
    admission, event = admission_for(proposal, 2)
    return system, system.adopt(proposal.goal_id, admission, event)


def test_ingestion_retains_only_proposal_and_exact_replay_is_idempotent() -> None:
    system = GoalSystem()
    proposal = make_proposal(origin_kind=R13ReferenceKind.OPERATOR_PROPOSAL)
    event = make_event(
        1,
        proposal.evidence_refs,
        event_id=proposal.revision_history[0].event_id,
        recorded_at=proposal.revision_history[0].created_at,
    )

    stored = system.ingest_proposal(proposal, event)
    first_snapshot = system.export()
    replayed = system.ingest_proposal(proposal, event)

    assert stored.lifecycle is GoalLifecycle.PROPOSED
    assert stored.subject_admission is None
    assert stored.subject_transition_proofs == ()
    assert replayed is stored
    assert system.snapshot().authority_digest == first_snapshot.authority_digest
    assert len(system.snapshot().event_receipts) == 1
    assert not hasattr(system, "complete")
    assert not hasattr(system, "fail")


@pytest.mark.parametrize(
    "origin_kind",
    [
        R13ReferenceKind.USER_REQUEST,
        R13ReferenceKind.OPERATOR_PROPOSAL,
        R13ReferenceKind.EXTERNAL_REQUEST,
        R13ReferenceKind.MOTIVATION,
    ],
)
def test_request_operator_external_and_motivation_origins_never_auto_adopt(
    origin_kind: R13ReferenceKind,
) -> None:
    system = GoalSystem()
    proposal = make_proposal(origin_kind=origin_kind)

    stored = ingest(system, proposal, 1)

    assert stored.lifecycle is GoalLifecycle.PROPOSED
    assert stored.subject_admission is None
    assert stored.subject_transition_proofs == ()


def test_repeated_external_request_does_not_turn_proposal_into_adoption() -> None:
    system = GoalSystem()
    proposal = make_proposal(origin_kind=R13ReferenceKind.EXTERNAL_REQUEST)
    first = ingest(system, proposal, 1)
    original_digest = first.record_digest

    repeated = ingest(system, proposal, 2, event_id="event:external:repeat")

    assert repeated.lifecycle is GoalLifecycle.PROPOSED
    assert repeated.record_digest == original_digest
    assert len(system.snapshot().event_receipts) == 2
    assert system.get(proposal.goal_id) == first


def test_same_goal_identity_with_different_proposal_contract_fails_atomically() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    ingest(system, proposal, 1)
    before = system.snapshot()
    collision = make_proposal(description="a conflicting contract")

    with pytest.raises(GoalSystemConflict, match="different proposal contract"):
        ingest(system, collision, 2, event_id="event:proposal:collision")

    assert system.snapshot() == before


def test_proposal_genesis_must_bind_exact_typed_goal_evidence() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    bad_genesis = replace(
        proposal.revision_history[0],
        evidence_refs=("event:forged-genesis-evidence",),
    )
    forged = replace(proposal, revision_history=(bad_genesis,))
    before = system.snapshot()

    with pytest.raises(GoalSystemError, match="genesis must bind exact Goal evidence"):
        ingest(system, forged, 1)

    assert system.snapshot() == before


def test_same_sequence_ingestion_requires_exact_genesis_time() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    before = system.snapshot()

    with pytest.raises(GoalSystemError, match="genesis event identity"):
        system.ingest_proposal(
            proposal,
            make_event(
                1,
                proposal.evidence_refs,
                event_id=proposal.revision_history[0].event_id,
                recorded_at=BASE_TIME + timedelta(microseconds=1),
            ),
        )

    assert system.snapshot() == before


def test_genesis_event_id_cannot_be_reused_at_a_different_sequence() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    before = system.snapshot()

    with pytest.raises(GoalSystemError, match="genesis event identity must match exactly"):
        system.ingest_proposal(
            proposal,
            make_event(
                2,
                proposal.evidence_refs,
                event_id=proposal.revision_history[0].event_id,
                recorded_at=BASE_TIME + timedelta(seconds=2),
            ),
        )

    assert system.snapshot() == before


def test_only_exact_caller_supplied_admission_can_adopt() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    ingest(system, proposal, 1)
    admission, event = admission_for(proposal, 2)
    before = system.snapshot()
    wrong_admission = replace(admission, proposal_digest="0" * 64)

    with pytest.raises(TypeError, match="admission must be GoalSubjectAdmission"):
        system.adopt(proposal.goal_id, None, event)  # type: ignore[arg-type]
    assert system.snapshot() == before
    with pytest.raises(GoalSystemError, match="does not match exact proposal/event"):
        system.adopt(proposal.goal_id, wrong_admission, event)

    assert system.snapshot() == before
    adopted = system.adopt(proposal.goal_id, admission, event)
    after_adoption = system.snapshot()
    replayed = system.adopt(proposal.goal_id, admission, event)

    assert adopted.lifecycle is GoalLifecycle.ADOPTED
    assert adopted.subject_admission == admission
    assert replayed is adopted
    assert system.snapshot().authority_digest == after_adoption.authority_digest
    assert not hasattr(system, "create_admission")


def test_event_identity_cannot_be_reused_with_conflicting_operation_input() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    event = make_event(
        1,
        proposal.evidence_refs,
        event_id=proposal.revision_history[0].event_id,
        recorded_at=proposal.revision_history[0].created_at,
    )
    system.ingest_proposal(proposal, event)
    before = system.snapshot()
    conflicting = make_event(
        1,
        (R13Reference(R13ReferenceKind.EVENT, "event:other:evidence"),),
        event_id=event.event_id,
        recorded_at=event.recorded_at,
    )

    with pytest.raises(GoalSystemConflict, match="reused with conflicting input"):
        system.ingest_proposal(proposal, conflicting)

    assert system.snapshot() == before


def test_subject_defer_resume_and_abandon_need_fresh_single_use_proofs() -> None:
    system, adopted = adopted_system()
    defer_proof, defer_event = transition_for(
        adopted,
        3,
        GoalRevisionOperation.DEFER,
        GoalRevisionReason.SUBJECT_DEFERRED,
    )
    deferred = system.defer(adopted.goal_id, defer_proof, defer_event)
    deferred_snapshot = system.snapshot()
    assert deferred.lifecycle is GoalLifecycle.DEFERRED
    assert system.defer(adopted.goal_id, defer_proof, defer_event) is deferred
    assert system.snapshot().authority_digest == deferred_snapshot.authority_digest

    stale_event = make_event(
        4,
        (R13Reference(R13ReferenceKind.EVENT, "event:decision:4"),),
    )
    with pytest.raises(GoalSystemError, match="does not match exact Goal/event evidence"):
        system.resume(adopted.goal_id, defer_proof, stale_event)
    assert system.snapshot() == deferred_snapshot

    resume_proof, resume_event = transition_for(
        deferred,
        4,
        GoalRevisionOperation.ADOPT,
        GoalRevisionReason.SUBJECT_ADMISSION,
    )
    resumed = system.resume(adopted.goal_id, resume_proof, resume_event)
    assert resumed.lifecycle is GoalLifecycle.ADOPTED

    abandon_proof, abandon_event = transition_for(
        resumed,
        5,
        GoalRevisionOperation.ABANDON,
        GoalRevisionReason.SUBJECT_ABANDONED,
    )
    abandoned = system.abandon(adopted.goal_id, abandon_proof, abandon_event)
    assert abandoned.lifecycle is GoalLifecycle.ABANDONED

    reopen_event = make_event(
        6,
        (R13Reference(R13ReferenceKind.EVENT, "event:decision:6"),),
    )
    reopen_proof = GoalSubjectTransitionProof(
        goal_id=abandoned.goal_id,
        proposal_digest=abandoned.proposal_digest,
        operation=GoalRevisionOperation.ADOPT,
        reason=GoalRevisionReason.SUBJECT_ADMISSION,
        previous_lifecycle_state=GoalLifecycle.DEFERRED,
        evidence_refs=tuple(item.reference for item in reopen_event.evidence_refs),
        event_id=reopen_event.event_id,
        event_sequence=reopen_event.event_sequence,
    )
    before_reopen = system.snapshot()
    with pytest.raises(GoalSystemError, match="does not allow this subject transition"):
        system.resume(adopted.goal_id, reopen_proof, reopen_event)
    assert system.snapshot() == before_reopen


def test_transition_rejects_wrong_goal_operation_prior_lifecycle_and_stale_admission() -> None:
    system, adopted = adopted_system()
    wrong_operation, event = transition_for(
        adopted,
        3,
        GoalRevisionOperation.ABANDON,
        GoalRevisionReason.SUBJECT_ABANDONED,
    )
    before = system.snapshot()
    with pytest.raises(GoalSystemError, match="does not match exact Goal/event evidence"):
        system.defer(adopted.goal_id, wrong_operation, event)

    wrong_prior = GoalSubjectTransitionProof(
        goal_id=adopted.goal_id,
        proposal_digest=adopted.proposal_digest,
        operation=GoalRevisionOperation.DEFER,
        reason=GoalRevisionReason.SUBJECT_DEFERRED,
        previous_lifecycle_state=GoalLifecycle.DEFERRED,
        evidence_refs=tuple(item.reference for item in event.evidence_refs),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
    )
    with pytest.raises(GoalSystemError, match="stale prior lifecycle"):
        system.defer(adopted.goal_id, wrong_prior, event)
    with pytest.raises(TypeError, match="proof must be GoalSubjectTransitionProof"):
        system.defer(adopted.goal_id, adopted.subject_admission, event)  # type: ignore[arg-type]

    wrong_proposal = GoalSubjectTransitionProof(
        goal_id=adopted.goal_id,
        proposal_digest="0" * 64,
        operation=GoalRevisionOperation.DEFER,
        reason=GoalRevisionReason.SUBJECT_DEFERRED,
        previous_lifecycle_state=GoalLifecycle.ADOPTED,
        evidence_refs=tuple(item.reference for item in event.evidence_refs),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
    )
    with pytest.raises(GoalSystemError, match="does not match exact Goal/event evidence"):
        system.defer(adopted.goal_id, wrong_proposal, event)

    other = make_proposal(key="other")
    wrong_goal = GoalSubjectTransitionProof(
        goal_id=other.goal_id,
        proposal_digest=adopted.proposal_digest,
        operation=GoalRevisionOperation.DEFER,
        reason=GoalRevisionReason.SUBJECT_DEFERRED,
        previous_lifecycle_state=GoalLifecycle.ADOPTED,
        evidence_refs=tuple(item.reference for item in event.evidence_refs),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
    )
    with pytest.raises(GoalSystemError, match="does not match exact Goal/event evidence"):
        system.defer(adopted.goal_id, wrong_goal, event)
    assert system.snapshot() == before


def test_conflicting_adopted_goals_coexist_without_ranking_or_suspension() -> None:
    system = GoalSystem()
    first = make_proposal(key="health")
    second = make_proposal(
        key="fitness",
        conflicts=(first.goal_id,),
        genesis_event_id="event:goal:second-genesis",
        genesis_sequence=2,
        created_at=BASE_TIME + timedelta(seconds=2),
    )
    ingest(system, first, 1)
    ingest(system, second, 2)

    first_admission, first_event = admission_for(first, 3)
    active_first = system.adopt(first.goal_id, first_admission, first_event)
    second_admission, second_event = admission_for(second, 4)
    active_second = system.adopt(second.goal_id, second_admission, second_event)

    assert active_first.lifecycle is GoalLifecycle.ADOPTED
    assert active_second.lifecycle is GoalLifecycle.ADOPTED
    assert active_second.conflicts == (first.goal_id,)
    assert {item.goal_id for item in system.records} == {first.goal_id, second.goal_id}


def test_missing_dependency_and_dependency_cycle_fail_closed() -> None:
    system = GoalSystem()
    missing = make_proposal(dependencies=("goal:missing",))
    before = system.snapshot()
    with pytest.raises(GoalSystemConflict, match="dependency/conflict facts are invalid"):
        ingest(system, missing, 1)
    assert system.snapshot() == before

    dangling_conflict = make_proposal(conflicts=("goal:missing-conflict",))
    with pytest.raises(GoalSystemConflict, match="dependency/conflict facts are invalid"):
        ingest(system, dangling_conflict, 1)
    assert system.snapshot() == before

    target_a = R13Reference(R13ReferenceKind.VALUE, "value:cycle-a")
    target_b = R13Reference(R13ReferenceKind.VALUE, "value:cycle-b")
    goal_a_id = goal_id_for_target(target_a)
    goal_b_id = goal_id_for_target(target_b)
    cycle_a = make_proposal(key="cycle-a", dependencies=(goal_b_id,))
    cycle_b = make_proposal(
        key="cycle-b",
        dependencies=(goal_a_id,),
        genesis_event_id="event:cycle:b",
        genesis_sequence=2,
        created_at=BASE_TIME + timedelta(seconds=2),
    )
    cycle_records = tuple(sorted((cycle_a, cycle_b), key=lambda item: item.goal_id))
    with pytest.raises(GoalSystemError, match="Goal graph is invalid"):
        GoalSystemSnapshot(records=cycle_records)


def test_sequence_and_timestamp_regressions_are_atomic() -> None:
    system = GoalSystem()
    first = make_proposal()
    ingest(system, first, 3, event_id="event:proposal:sequence-three")
    before = system.snapshot()
    lower = make_proposal(
        key="second",
        genesis_event_id="event:goal:second-genesis",
        genesis_sequence=1,
    )
    with pytest.raises(GoalSystemError, match="sequence regressed"):
        ingest(system, lower, 2, event_id="event:proposal:sequence-two")
    assert system.snapshot() == before

    time_regression = make_event(
        4,
        first.evidence_refs,
        event_id="event:proposal:time-regression",
        recorded_at=BASE_TIME + timedelta(seconds=2),
    )
    with pytest.raises(GoalSystemError, match="time regressed"):
        system.ingest_proposal(first, time_regression)
    assert system.snapshot() == before


def test_event_identity_and_per_event_evidence_bounds_are_closed() -> None:
    refs = tuple(
        sorted(
            (
                R13Reference(R13ReferenceKind.EVENT, f"event:bound:{index:02d}")
                for index in range(goal_system_module.GOAL_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT)
            ),
            key=lambda item: (item.reference, item.kind.value),
        )
    )
    valid = GoalMutationEvidence(
        event_id="event:bound:maximum",
        event_sequence=R13_MAX_EVENT_SEQUENCE,
        recorded_at=BASE_TIME,
        evidence_refs=refs,
    )
    assert len(valid.evidence_refs) == goal_system_module.GOAL_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT
    with pytest.raises(ValueError, match="event_sequence must be a positive bounded integer"):
        GoalMutationEvidence(
            event_id="event:bound:overflow",
            event_sequence=R13_MAX_EVENT_SEQUENCE + 1,
            recorded_at=BASE_TIME,
            evidence_refs=(
                R13Reference(R13ReferenceKind.EVENT, "event:bound:overflow"),
            ),
        )

    too_many = tuple(
        sorted(
            (
                R13Reference(R13ReferenceKind.EVENT, f"event:overflow:{index:02d}")
                for index in range(goal_system_module.GOAL_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT + 1)
            ),
            key=lambda item: (item.reference, item.kind.value),
        )
    )
    with pytest.raises(ValueError, match="evidence_refs exceeds its bound"):
        make_event(1, too_many)


def test_snapshot_rejects_missing_receipts_and_revision_timestamp_mismatch() -> None:
    system, adopted = adopted_system()
    proof, event = transition_for(
        adopted,
        3,
        GoalRevisionOperation.DEFER,
        GoalRevisionReason.SUBJECT_DEFERRED,
    )
    deferred = system.defer(adopted.goal_id, proof, event)
    snapshot = system.snapshot()

    incomplete = tuple(
        receipt
        for receipt in snapshot.event_receipts
        if receipt.operation is not GoalSystemEventOperation.ADOPT
    )
    with pytest.raises(ValueError, match="revision count differs"):
        GoalSystemSnapshot(records=snapshot.records, event_receipts=incomplete)

    altered_receipts = tuple(
        replace(
            receipt,
            recorded_at=receipt.recorded_at + timedelta(seconds=1),
            revision_witness=replace(
                receipt.revision_witness,
                created_at=receipt.revision_witness.created_at + timedelta(seconds=1),
            ),
        )
        if receipt.event_id == event.event_id
        else receipt
        for receipt in snapshot.event_receipts
    )
    with pytest.raises(ValueError, match="differs from its exact mutation revision prefix"):
        GoalSystemSnapshot(records=snapshot.records, event_receipts=altered_receipts)

    assert deferred.lifecycle is GoalLifecycle.DEFERRED


def test_snapshot_binds_ingestion_receipt_to_retained_genesis_record() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    ingest(system, proposal, 1)
    snapshot = system.snapshot()
    receipt = snapshot.event_receipts[0]
    forged_digest = "f" * 64
    forged_input = goal_system_module._goal_input_digest(
        GoalSystemEventOperation.INGEST_PROPOSAL,
        goal_id=receipt.goal_id,
        proposal_digest=receipt.proposal_digest,
        evidence_refs=receipt.evidence_refs,
        proposal_record_digest=forged_digest,
    )
    forged_receipt = replace(
        receipt,
        proposal_record_digest=forged_digest,
        input_digest=forged_input,
    )

    with pytest.raises(ValueError, match="does not bind the exact Goal genesis record"):
        GoalSystemSnapshot(
            records=snapshot.records,
            event_receipts=(forged_receipt,),
        )


def test_subject_proofs_and_receipts_survive_bounded_revision_compaction() -> None:
    system, record = adopted_system()
    for sequence in range(3, 13):
        if record.lifecycle is GoalLifecycle.ADOPTED:
            proof, event = transition_for(
                record,
                sequence,
                GoalRevisionOperation.DEFER,
                GoalRevisionReason.SUBJECT_DEFERRED,
            )
            record = system.defer(record.goal_id, proof, event)
        else:
            proof, event = transition_for(
                record,
                sequence,
                GoalRevisionOperation.ADOPT,
                GoalRevisionReason.SUBJECT_ADMISSION,
            )
            record = system.resume(record.goal_id, proof, event)

    snapshot = system.snapshot()
    assert record.history_anchor is not None
    assert record.history_anchor.through_revision > 0
    assert len(record.revision_history) <= goal_system_module.R13_MAX_REVISION_HISTORY
    assert len(snapshot.event_receipts) == record.revision + 1
    assert snapshot == system.export()

    compacted_receipt = next(
        item
        for item in snapshot.event_receipts
        if item.event_sequence == 3
        and item.operation is GoalSystemEventOperation.DEFER
    )
    assert compacted_receipt.revision_witness is not None
    old_revision = compacted_receipt.revision_witness
    forged_proof = GoalSubjectTransitionProof(
        goal_id=old_revision.goal_id,
        proposal_digest=old_revision.proposal_digest,
        operation=GoalRevisionOperation.ABANDON,
        reason=GoalRevisionReason.SUBJECT_ABANDONED,
        previous_lifecycle_state=GoalLifecycle.ADOPTED,
        evidence_refs=tuple(item.reference for item in compacted_receipt.evidence_refs),
        event_id=compacted_receipt.event_id,
        event_sequence=compacted_receipt.event_sequence,
    )
    forged_revision = replace(
        old_revision,
        operation=GoalRevisionOperation.ABANDON,
        reason=GoalRevisionReason.SUBJECT_ABANDONED,
        evidence_refs=tuple(
            sorted((*forged_proof.evidence_refs, forged_proof.transition_digest))
        ),
    )
    forged_input_digest = goal_system_module._goal_input_digest(
        GoalSystemEventOperation.ABANDON,
        goal_id=compacted_receipt.goal_id,
        proposal_digest=compacted_receipt.proposal_digest,
        evidence_refs=compacted_receipt.evidence_refs,
        transition_digest=forged_proof.transition_digest,
    )
    forged_receipt = replace(
        compacted_receipt,
        operation=GoalSystemEventOperation.ABANDON,
        input_digest=forged_input_digest,
        transition_digest=forged_proof.transition_digest,
        revision_witness=forged_revision,
    )
    forged_receipts = tuple(
        forged_receipt if item is compacted_receipt else item
        for item in snapshot.event_receipts
    )
    with pytest.raises(ValueError, match="breaks the exact revision chain"):
        GoalSystemSnapshot(records=snapshot.records, event_receipts=forged_receipts)


def test_record_and_receipt_capacity_fail_without_partial_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = GoalSystem()
    first = make_proposal()
    monkeypatch.setattr(goal_system_module, "GOAL_SYSTEM_MAX_RECORDS", 1)
    ingest(system, first, 1)
    before = system.snapshot()
    second = make_proposal(
        key="second",
        genesis_event_id="event:goal:second-genesis",
        genesis_sequence=2,
        created_at=BASE_TIME + timedelta(seconds=2),
    )
    with pytest.raises(GoalSystemCapacityExceeded, match="record capacity"):
        ingest(system, second, 2)
    assert system.snapshot() == before

    monkeypatch.setattr(goal_system_module, "GOAL_SYSTEM_MAX_RECORDS", 32)
    monkeypatch.setattr(goal_system_module, "GOAL_SYSTEM_MAX_EVENT_RECEIPTS", 2)
    ingest(system, first, 2, event_id="event:proposal:repeat")
    before_overflow = system.snapshot()
    with pytest.raises(GoalSystemCapacityExceeded, match="receipt capacity"):
        ingest(system, first, 3, event_id="event:proposal:overflow")
    assert system.snapshot() == before_overflow


def test_full_u1_goal_record_bound_is_preserved_and_next_record_fails() -> None:
    assert GOAL_SYSTEM_MAX_RECORDS == 32
    system = GoalSystem()
    for index in range(GOAL_SYSTEM_MAX_RECORDS):
        proposal = make_proposal(
            key=f"bounded-{index}",
            genesis_event_id=f"event:goal:genesis:{index}",
        )
        ingest(
            system,
            proposal,
            index + 1,
            event_id=(
                proposal.revision_history[0].event_id
                if index == 0
                else f"event:goal:ingest:{index}"
            ),
        )
    before = system.snapshot()
    overflow = make_proposal(
        key="bounded-overflow",
        genesis_event_id="event:goal:genesis:overflow",
    )

    with pytest.raises(GoalSystemCapacityExceeded, match="record capacity"):
        ingest(system, overflow, GOAL_SYSTEM_MAX_RECORDS + 1)

    assert len(system.records) == GOAL_SYSTEM_MAX_RECORDS
    assert system.snapshot() == before


def test_revision_and_snapshot_byte_capacity_fail_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, adopted = adopted_system()
    monkeypatch.setattr(goal_system_module, "R13_MAX_REVISION", 1)
    proof, event = transition_for(
        adopted,
        3,
        GoalRevisionOperation.DEFER,
        GoalRevisionReason.SUBJECT_DEFERRED,
    )
    before_revision_overflow = system.snapshot()
    with pytest.raises(GoalSystemCapacityExceeded, match="revision bound"):
        system.defer(adopted.goal_id, proof, event)
    assert system.snapshot() == before_revision_overflow

    monkeypatch.setattr(goal_system_module, "R13_MAX_REVISION", 2**31 - 1)
    small_limit = before_revision_overflow.serialized_bytes
    monkeypatch.setattr(goal_system_module, "GOAL_SYSTEM_MAX_SERIALIZED_BYTES", small_limit)
    repeated = make_event(
        3,
        adopted.evidence_refs,
        event_id="event:proposal:too-large",
    )
    with pytest.raises(GoalSystemCapacityExceeded, match="snapshot exceeds its byte bound"):
        system.ingest_proposal(make_proposal(), repeated)
    assert system.snapshot() == before_revision_overflow


def test_graph_conflict_facts_and_snapshot_export_are_exact_and_side_effect_free() -> None:
    system = GoalSystem()
    proposal = make_proposal()
    ingest(system, proposal, 1)
    before = system.snapshot()

    exported = system.export()
    system.validate()

    assert exported == before
    assert exported.authority_digest == system.snapshot().authority_digest
    assert exported.records[0].dependencies == ()
    assert exported.records[0].conflicts == ()
    assert not hasattr(system, "select_winner")
    assert not hasattr(system, "create_plan")
    assert not hasattr(system, "schedule")


def test_goal_system_module_has_no_later_runtime_model_or_scheduler_imports() -> None:
    source = Path(goal_system_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                imported_modules.append("<relative-import>")
            elif node.module is not None:
                imported_modules.append(node.module)

    forbidden_prefixes = (
        "suzka.runtime",
        "suzka.motivation.commitment",
        "suzka.decision",
        "suzka.attention",
        "suzka.relationship",
        "suzka.self_model",
        "suzka.action",
        "suzka.scheduler",
        "suzka.api",
        "suzka.models",
        "suzka.cognition",
        "suzka.persona",
        "suzka.tools",
        "torch",
        "transformers",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in forbidden_prefixes
    )
    allowed_project_imports = {
        "suzka.identifiers",
        "suzka.motivation.common",
        "suzka.motivation.goal",
    }
    assert all(
        module in allowed_project_imports
        or module.split(".", maxsplit=1)[0] in sys.stdlib_module_names
        for module in imported_modules
    )
    assert not any(
        hasattr(GoalSystem, name)
        for name in ("complete", "fail", "select_winner", "create_plan", "schedule")
    )


def test_public_exports_include_goal_system_authority() -> None:
    from suzka.motivation import GoalSystem as ExportedGoalSystem
    from suzka.motivation import GoalSystemSnapshot as ExportedGoalSystemSnapshot

    assert ExportedGoalSystem is GoalSystem
    assert ExportedGoalSystemSnapshot is GoalSystemSnapshot
    assert GOAL_SYSTEM_MAX_EVENT_RECEIPTS == 1_024
    assert GOAL_SYSTEM_MAX_RECORDS == 32
    assert GOAL_SYSTEM_MAX_SERIALIZED_BYTES == 32 * 1024 * 1024
