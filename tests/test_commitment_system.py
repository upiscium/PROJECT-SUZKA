"""Focused tests for bounded, process-local R13 U4 Commitment authority."""

import ast
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

import pytest

import suzka.motivation.commitment_system as commitment_system_module
from suzka.motivation.commitment import (
    CommitmentAdmissionReason,
    CommitmentLifecycle,
    CommitmentRecord,
    CommitmentRevisionOperation,
    CommitmentRevisionReason,
    CommitmentRevisionRecord,
    CommitmentSubjectAdmission,
    CommitmentSubjectTransitionProof,
    commitment_id_for_fields,
)
from suzka.motivation.commitment_system import (
    COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS,
    COMMITMENT_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT,
    COMMITMENT_SYSTEM_MAX_RECORDS,
    COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES,
    CommitmentMutationEvidence,
    CommitmentSystem,
    CommitmentSystemCapacityExceeded,
    CommitmentSystemConflict,
    CommitmentSystemError,
    CommitmentSystemEventOperation,
    CommitmentSystemSnapshot,
)
from suzka.motivation.common import (
    Deadline,
    R13_MAX_EVENT_SEQUENCE,
    R13Reference,
    R13ReferenceKind,
    RevisionCompactionAnchor,
)


BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def make_proposal(
    key: str = "report",
    *,
    beneficiary: R13Reference | None = None,
    scope: tuple[str, ...] | None = None,
    deadline: Deadline | None = None,
    origin_kind: R13ReferenceKind = R13ReferenceKind.OPERATOR_PROPOSAL,
    origin_refs: tuple[R13Reference, ...] | None = None,
    evidence_refs: tuple[R13Reference, ...] | None = None,
    related_goal_refs: tuple[R13Reference, ...] = (),
    desire_refs: tuple[R13Reference, ...] = (),
    genesis_id: str | None = None,
    genesis_sequence: int = 1,
    genesis_at: datetime = BASE_TIME,
) -> CommitmentRecord:
    subject = f"write {key}"
    owner = beneficiary or R13Reference(R13ReferenceKind.SUBJECT, "subject:self")
    terms = scope or (f"deliver {key}",)
    due = deadline or Deadline.without_deadline()
    evidence = evidence_refs or (
        R13Reference(R13ReferenceKind.EVENT, f"event:proposal:{key}"),
    )
    origins = origin_refs or (R13Reference(origin_kind, f"source:proposal:{key}"),)
    commitment_id = commitment_id_for_fields(
        subject=subject,
        beneficiary=owner,
        scope=terms,
        deadline=due,
        evidence_refs=evidence,
        origin_refs=origins,
        related_goal_refs=related_goal_refs,
        desire_refs=desire_refs,
    )
    genesis = CommitmentRevisionRecord(
        commitment_id,
        0,
        CommitmentRevisionOperation.CREATE,
        CommitmentRevisionReason.CREATION,
        genesis_at,
        None,
        event_id=genesis_id or f"event:genesis:{key}",
        event_sequence=genesis_sequence,
        evidence_refs=tuple(sorted(item.reference for item in evidence)),
    )
    return CommitmentRecord(
        subject=subject,
        beneficiary=owner,
        scope=terms,
        deadline=due,
        evidence_refs=evidence,
        origin_refs=origins,
        related_goal_refs=related_goal_refs,
        desire_refs=desire_refs,
        revision_history=(genesis,),
    )


def event_for(
    sequence: int,
    refs: tuple[R13Reference, ...],
    *,
    event_id: str | None = None,
    at: datetime | None = None,
) -> CommitmentMutationEvidence:
    return CommitmentMutationEvidence(
        event_id=event_id or f"event:mutation:{sequence}",
        event_sequence=sequence,
        recorded_at=at or BASE_TIME + timedelta(seconds=sequence),
        evidence_refs=refs,
    )


def ingest(
    system: CommitmentSystem,
    proposal: CommitmentRecord,
    sequence: int,
    *,
    event_id: str | None = None,
) -> CommitmentRecord:
    genesis = proposal.revision_history[0]
    if sequence == genesis.event_sequence:
        assert event_id is None or event_id == genesis.event_id
        receipt_event = event_for(
            sequence,
            proposal.evidence_refs,
            event_id=genesis.event_id,
            at=genesis.created_at,
        )
    else:
        receipt_event = event_for(
            sequence,
            proposal.evidence_refs,
            event_id=event_id,
        )
    return system.ingest_proposal(proposal, receipt_event)


def admission_for(
    proposal: CommitmentRecord,
    sequence: int,
    *,
    event_id: str | None = None,
) -> tuple[CommitmentSubjectAdmission, CommitmentMutationEvidence]:
    event = event_for(
        sequence,
        proposal.evidence_refs,
        event_id=event_id or f"event:accept:{sequence}",
    )
    return (
        CommitmentSubjectAdmission(
            proposal.commitment_id,
            proposal.proposal_digest,
            proposal.beneficiary,
            proposal.scope,
            proposal.deadline,
            tuple(item.reference for item in proposal.evidence_refs),
            event.event_id,
            event.event_sequence,
            CommitmentAdmissionReason.SUBJECT_RESPONSIBILITY,
        ),
        event,
    )


def transition_for(
    current: CommitmentRecord,
    sequence: int,
    operation: CommitmentRevisionOperation,
    *,
    refs: tuple[R13Reference, ...] | None = None,
) -> tuple[CommitmentSubjectTransitionProof, CommitmentMutationEvidence]:
    evidence = refs or (
        R13Reference(R13ReferenceKind.EVENT, f"event:decision:{sequence}"),
    )
    event = event_for(sequence, evidence)
    reason = {
        CommitmentRevisionOperation.RELEASE: CommitmentRevisionReason.SUBJECT_RELEASE,
        CommitmentRevisionOperation.RENEGOTIATE: (
            CommitmentRevisionReason.SUBJECT_RENEGOTIATION
        ),
    }[operation]
    return (
        CommitmentSubjectTransitionProof(
            current.commitment_id,
            current.proposal_digest,
            current.beneficiary,
            current.scope,
            current.deadline,
            operation,
            reason,
            CommitmentLifecycle.ACTIVE,
            tuple(item.reference for item in evidence),
            event.event_id,
            event.event_sequence,
        ),
        event,
    )


def active_system(
    *,
    key: str = "report",
    proposal: CommitmentRecord | None = None,
) -> tuple[CommitmentSystem, CommitmentRecord]:
    system = CommitmentSystem()
    original = proposal or make_proposal(key)
    ingest(system, original, 1)
    admission, event = admission_for(original, 2)
    return system, system.accept(original.commitment_id, admission, event)


@pytest.mark.parametrize(
    "origin_kind",
    [
        R13ReferenceKind.OPERATOR_PROPOSAL,
        R13ReferenceKind.USER_REQUEST,
        R13ReferenceKind.EXTERNAL_REQUEST,
        R13ReferenceKind.GOAL,
    ],
)
def test_proposal_sources_cannot_activate_responsibility(
    origin_kind: R13ReferenceKind,
) -> None:
    system = CommitmentSystem()
    proposal = make_proposal(origin_kind=origin_kind)
    original = ingest(system, proposal, 1)
    before = system.snapshot()

    assert original.lifecycle is CommitmentLifecycle.PROPOSED
    assert original.subject_admission is None
    assert original.outcome_evidence_refs == ()
    assert system.ingest_proposal(
        proposal,
        event_for(1, proposal.evidence_refs, event_id="event:genesis:report", at=BASE_TIME),
    ) is original
    assert system.snapshot() == before
    assert ingest(system, proposal, 2, event_id="event:repeat:request") is original
    assert system.get(proposal.commitment_id) is original
    assert system.records == (original,)
    assert system.snapshot().records[0].lifecycle is CommitmentLifecycle.PROPOSED


def test_accept_consumes_exact_supplied_proof_without_producing_one() -> None:
    system = CommitmentSystem()
    proposal = make_proposal(
        deadline=Deadline(BASE_TIME + timedelta(days=1)),
        beneficiary=R13Reference(R13ReferenceKind.EXTERNAL_PARTY, "party:client"),
        scope=("deliver report", "verify report"),
    )
    ingest(system, proposal, 1)
    admission, event = admission_for(proposal, 2)
    before = system.snapshot()
    with pytest.raises(TypeError, match="CommitmentSubjectAdmission"):
        system.accept(proposal.commitment_id, None, event)  # type: ignore[arg-type]
    assert system.snapshot() == before

    active = system.accept(proposal.commitment_id, admission, event)
    snapshot = system.snapshot()
    assert active.lifecycle is CommitmentLifecycle.ACTIVE
    assert active.commitment_id == proposal.commitment_id
    assert active.subject_admission is admission
    assert active.scope == proposal.scope
    assert active.deadline == proposal.deadline
    assert active.revision_history[-1].evidence_refs == tuple(
        sorted((*admission.evidence_refs, admission.admission_digest))
    )
    assert snapshot.canonical_value()["records"][0]["admission"]["scope"] == list(
        proposal.scope
    )
    assert snapshot.canonical_value()["event_receipts"][1]["admission"][
        "admission_digest"
    ] == admission.admission_digest
    assert system.accept(proposal.commitment_id, admission, event) is active
    assert system.snapshot() == snapshot
    assert not hasattr(system, "create_admission")


@pytest.mark.parametrize(
    "change",
    [
        {"beneficiary": R13Reference(R13ReferenceKind.EXTERNAL_PARTY, "party:other")},
        {"scope": ("different scope",)},
        {"deadline": Deadline(BASE_TIME + timedelta(days=4))},
        {"evidence_refs": ("event:other:evidence",)},
        {"event_id": "event:wrong-admission"},
        {"event_sequence": 3},
    ],
)
def test_wrong_admission_shape_or_event_fails_atomically(change: dict[str, object]) -> None:
    proposal = make_proposal()
    system = CommitmentSystem()
    ingest(system, proposal, 1)
    admission, event = admission_for(proposal, 2)
    before = system.snapshot()

    with pytest.raises(CommitmentSystemError, match="does not match"):
        system.accept(proposal.commitment_id, replace(admission, **change), event)
    assert system.snapshot() == before


def test_wrong_commitment_and_event_evidence_fail_without_mutation() -> None:
    system = CommitmentSystem()
    proposal = make_proposal()
    ingest(system, proposal, 1)
    other = make_proposal("other")
    wrong_admission, event = admission_for(other, 2)
    before = system.snapshot()
    with pytest.raises(CommitmentSystemError, match="does not match"):
        system.accept(proposal.commitment_id, wrong_admission, event)
    admission, valid_event = admission_for(proposal, 2)
    wrong_evidence = event_for(
        2,
        (R13Reference(R13ReferenceKind.EVENT, "event:wrong:evidence"),),
        event_id=valid_event.event_id,
    )
    with pytest.raises(CommitmentSystemError, match="does not match"):
        system.accept(proposal.commitment_id, admission, wrong_evidence)
    assert system.snapshot() == before


def test_proposal_genesis_must_bind_exact_evidence_and_advanced_input_is_rejected() -> None:
    system = CommitmentSystem()
    proposal = make_proposal()
    forged = replace(
        proposal,
        revision_history=(
            replace(
                proposal.revision_history[0],
                evidence_refs=("event:not-proposal-evidence",),
            ),
        ),
    )
    before = system.snapshot()
    with pytest.raises(CommitmentSystemError, match="genesis must bind exact"):
        ingest(system, forged, 1)
    assert system.snapshot() == before

    accepted_system, accepted = active_system()
    accepted_before = accepted_system.snapshot()
    with pytest.raises(CommitmentSystemError, match="proposal state only"):
        accepted_system.ingest_proposal(
            accepted,
            event_for(3, accepted.evidence_refs, event_id="event:advanced:input"),
        )
    assert accepted_system.snapshot() == accepted_before


@pytest.mark.parametrize(
    ("operation", "next_state"),
    [
        (CommitmentRevisionOperation.RELEASE, CommitmentLifecycle.RELEASED),
        (CommitmentRevisionOperation.RENEGOTIATE, CommitmentLifecycle.RENEGOTIATED),
    ],
)
def test_fresh_transition_proofs_release_or_renegotiate_without_reopening(
    operation: CommitmentRevisionOperation,
    next_state: CommitmentLifecycle,
) -> None:
    system, active = active_system()
    proof, event = transition_for(active, 3, operation)
    before = system.snapshot()
    with pytest.raises(TypeError, match="CommitmentSubjectTransitionProof"):
        system.release(active.commitment_id, active.subject_admission, event)  # type: ignore[arg-type]
    assert system.snapshot() == before

    apply_transition = (
        system.release if operation is CommitmentRevisionOperation.RELEASE else system.renegotiate
    )
    terminal = apply_transition(active.commitment_id, proof, event)
    snapshot = system.snapshot()
    assert terminal.lifecycle is next_state
    assert terminal.commitment_id == active.commitment_id
    assert terminal.scope == active.scope
    assert terminal.subject_transition_proofs == (proof,)
    assert terminal.revision_history[-1].evidence_refs == tuple(
        sorted((*proof.evidence_refs, proof.transition_digest))
    )
    assert snapshot.canonical_value()["event_receipts"][-1]["transition_proof"][
        "transition_digest"
    ] == proof.transition_digest
    assert apply_transition(active.commitment_id, proof, event) is terminal
    assert system.snapshot() == snapshot

    with pytest.raises(CommitmentSystemError, match="does not match"):
        apply_transition(
            active.commitment_id,
            proof,
            event_for(4, event.evidence_refs),
        )
    assert system.snapshot() == snapshot

    new_proof, new_event = transition_for(active, 4, operation)
    with pytest.raises(CommitmentSystemError, match="only active"):
        apply_transition(active.commitment_id, new_proof, new_event)
    with pytest.raises(CommitmentSystemError, match="only proposed"):
        new_admission, admission_event = admission_for(active, 4)
        system.accept(active.commitment_id, new_admission, admission_event)
    assert system.snapshot() == snapshot

    proposal = make_proposal()
    repeated = ingest(system, proposal, 5, event_id="event:repeat:terminal")
    assert repeated is terminal
    assert system.get(active.commitment_id) is terminal


def test_transition_rejects_wrong_authority_terms_operation_evidence_and_event() -> None:
    system, active = active_system()
    proof, event = transition_for(active, 3, CommitmentRevisionOperation.RELEASE)
    wrongs = (
        replace(proof, commitment_id=make_proposal("other").commitment_id,
                proposal_digest=make_proposal("other").proposal_digest),
        replace(proof, beneficiary=R13Reference(R13ReferenceKind.EXTERNAL_PARTY, "party:other")),
        replace(proof, scope=("different scope",)),
        replace(proof, deadline=Deadline(BASE_TIME + timedelta(days=1))),
        replace(proof, evidence_refs=("event:different",)),
        replace(proof, event_id="event:other"),
        replace(proof, event_sequence=4),
        transition_for(active, 3, CommitmentRevisionOperation.RENEGOTIATE)[0],
    )
    before = system.snapshot()
    for invalid in wrongs:
        with pytest.raises(CommitmentSystemError, match="does not match"):
            system.release(active.commitment_id, invalid, event)
        assert system.snapshot() == before

    wrong_event = event_for(
        3,
        (R13Reference(R13ReferenceKind.EVENT, "event:different"),),
        event_id=event.event_id,
    )
    with pytest.raises(CommitmentSystemError, match="does not match"):
        system.release(active.commitment_id, proof, wrong_event)
    assert system.snapshot() == before


def test_desire_goal_and_deadline_are_references_not_automatic_authority() -> None:
    proposal = make_proposal(
        deadline=Deadline(BASE_TIME + timedelta(seconds=1)),
        desire_refs=(R13Reference(R13ReferenceKind.MOTIVATION, "motivation:vanished"),),
        related_goal_refs=(R13Reference(R13ReferenceKind.GOAL, "goal:abandoned"),),
    )
    system, active = active_system(proposal=proposal)
    late_request = event_for(3, proposal.evidence_refs, at=BASE_TIME + timedelta(days=300))
    repeated = system.ingest_proposal(proposal, late_request)
    assert repeated is active
    assert repeated.lifecycle is CommitmentLifecycle.ACTIVE
    assert repeated.desire_refs == proposal.desire_refs
    assert repeated.related_goal_refs == proposal.related_goal_refs
    for name in ("fulfill", "breach", "expire", "tick", "auto_release", "schedule"):
        assert not hasattr(system, name)


def test_snapshot_rejects_subject_mutation_before_first_ingestion() -> None:
    system, _ = active_system()
    snapshot = system.snapshot()
    ingestion, acceptance = snapshot.event_receipts
    late_ingestion = replace(
        ingestion,
        event_id="event:late:ingestion",
        event_sequence=3,
        recorded_at=BASE_TIME + timedelta(seconds=3),
    )
    with pytest.raises(ValueError, match="precedes proposal ingestion"):
        CommitmentSystemSnapshot(
            records=snapshot.records,
            event_receipts=(acceptance, late_ingestion),
        )
    assert system.snapshot() == snapshot


def test_snapshot_rejects_release_before_first_ingestion() -> None:
    system, active = active_system()
    proof, event = transition_for(active, 3, CommitmentRevisionOperation.RELEASE)
    system.release(active.commitment_id, proof, event)
    snapshot = system.snapshot()
    ingestion, acceptance, release = snapshot.event_receipts
    late_ingestion = replace(
        ingestion,
        event_id="event:late:after-release",
        event_sequence=4,
        recorded_at=BASE_TIME + timedelta(seconds=4),
    )
    with pytest.raises(ValueError, match="precedes proposal ingestion"):
        CommitmentSystemSnapshot(
            records=snapshot.records,
            event_receipts=(acceptance, release, late_ingestion),
        )
    assert system.snapshot() == snapshot


def test_delayed_initial_ingestion_later_repeat_and_replay() -> None:
    system = CommitmentSystem()
    proposal = make_proposal()
    ingest(system, proposal, 2, event_id="event:delayed:ingest")
    admission, event = admission_for(proposal, 3)
    accepted = system.accept(proposal.commitment_id, admission, event)
    repeat_event = event_for(4, proposal.evidence_refs, event_id="event:repeat:active")
    assert system.ingest_proposal(proposal, repeat_event) is accepted
    before = system.snapshot()
    assert tuple(item.operation for item in before.event_receipts) == (
        CommitmentSystemEventOperation.INGEST_PROPOSAL,
        CommitmentSystemEventOperation.ACCEPT,
        CommitmentSystemEventOperation.INGEST_PROPOSAL,
    )
    assert system.ingest_proposal(proposal, repeat_event) is accepted
    assert system.accept(proposal.commitment_id, admission, event) is accepted
    assert system.snapshot() == before


def test_shared_upstream_genesis_event_may_propose_distinct_responsibilities() -> None:
    system = CommitmentSystem()
    first = make_proposal("one", genesis_id="event:shared:genesis")
    second = make_proposal("two", genesis_id="event:shared:genesis")
    ingest(system, first, 2, event_id="event:ingest:first")
    ingest(system, second, 3, event_id="event:ingest:second")
    assert len(system.snapshot().records) == 2
    assert first.commitment_id != second.commitment_id


@pytest.mark.parametrize(
    ("second_id", "second_sequence", "second_at"),
    [
        ("event:shared:genesis", 2, BASE_TIME + timedelta(seconds=2)),
        ("event:other:genesis", 1, BASE_TIME),
        ("event:shared:genesis", 1, BASE_TIME + timedelta(microseconds=1)),
    ],
)
def test_conflicting_shared_genesis_identity_fails_without_partial_ingestion(
    second_id: str, second_sequence: int, second_at: datetime
) -> None:
    system = CommitmentSystem()
    first = make_proposal("one", genesis_id="event:shared:genesis")
    second = make_proposal(
        "two",
        genesis_id=second_id,
        genesis_sequence=second_sequence,
        genesis_at=second_at,
    )
    ingest(system, first, 3, event_id="event:ingest:first")
    before = system.snapshot()

    with pytest.raises(ValueError, match="genesis event identity conflicts"):
        ingest(system, second, 4, event_id="event:ingest:second")

    assert system.snapshot() == before


@pytest.mark.parametrize(
    ("second_id", "second_sequence", "second_at"),
    [
        ("event:cross:receipt", 3, BASE_TIME + timedelta(seconds=3)),
        ("event:other:genesis", 2, BASE_TIME + timedelta(seconds=2)),
        ("event:cross:receipt", 2, BASE_TIME + timedelta(seconds=2, microseconds=1)),
    ],
)
def test_genesis_and_system_receipt_event_identity_must_agree(
    second_id: str, second_sequence: int, second_at: datetime
) -> None:
    system = CommitmentSystem()
    first = make_proposal("one", genesis_id="event:first:genesis")
    second = make_proposal(
        "two", genesis_id=second_id,
        genesis_sequence=second_sequence, genesis_at=second_at,
    )
    ingest(system, first, 2, event_id="event:cross:receipt")
    before = system.snapshot()

    with pytest.raises(ValueError, match="genesis event identity conflicts"):
        ingest(system, second, 4, event_id="event:ingest:second")

    assert system.snapshot() == before


def test_orphaned_receipt_and_missing_or_tampered_history_fail_closed() -> None:
    system = CommitmentSystem()
    proposal = make_proposal()
    ingest(system, proposal, 2, event_id="event:delayed:genesis-check")
    admission, event = admission_for(proposal, 3)
    active = system.accept(proposal.commitment_id, admission, event)
    snapshot = system.snapshot()
    with pytest.raises(ValueError, match="missing record"):
        CommitmentSystemSnapshot(records=(), event_receipts=snapshot.event_receipts)
    with pytest.raises(ValueError, match="revision count differs"):
        CommitmentSystemSnapshot(records=snapshot.records, event_receipts=snapshot.event_receipts[:1])

    forged_genesis = replace(
        snapshot.event_receipts[0].proposal_genesis,
        created_at=BASE_TIME + timedelta(microseconds=1),
    )
    forged_receipt = replace(snapshot.event_receipts[0], proposal_genesis=forged_genesis)
    with pytest.raises(ValueError, match="genesis"):
        CommitmentSystemSnapshot(
            records=snapshot.records,
            event_receipts=(forged_receipt, *snapshot.event_receipts[1:]),
        )
    assert active.lifecycle is CommitmentLifecycle.ACTIVE
    assert system.snapshot() == snapshot


def test_unreachable_compaction_and_unverified_outcome_records_are_rejected() -> None:
    system, active = active_system()
    snapshot = system.snapshot()
    genesis, admitted = active.revision_history
    anchor = RevisionCompactionAnchor(
        authority_id=active.commitment_id,
        through_revision=genesis.revision,
        through_digest=genesis.record_digest,
        through_created_at=genesis.created_at,
        through_evidence_refs=genesis.evidence_refs,
        through_previous_revision_digest=genesis.previous_revision_digest,
        through_state=CommitmentLifecycle.PROPOSED.value,
        through_previous_state=None,
        through_operation=CommitmentRevisionOperation.CREATE.value,
        through_reason=CommitmentRevisionReason.CREATION.value,
        through_event_id=genesis.event_id,
        through_event_sequence=genesis.event_sequence,
    )
    with pytest.raises(ValueError, match="full bounded suffix"):
        replace(active, revision_history=(admitted,), history_anchor=anchor)

    outcome = R13Reference(R13ReferenceKind.EVENT, "event:verified:outcome")
    terminal_revision = CommitmentRevisionRecord(
        active.commitment_id,
        2,
        CommitmentRevisionOperation.FULFILL,
        CommitmentRevisionReason.VERIFIED_OUTCOME,
        BASE_TIME + timedelta(seconds=3),
        CommitmentLifecycle.ACTIVE,
        event_id="event:terminal:outcome",
        event_sequence=3,
        evidence_refs=(outcome.reference,),
        previous_revision_digest=admitted.record_digest,
    )
    fulfilled = replace(
        active,
        lifecycle=CommitmentLifecycle.FULFILLED,
        revision=2,
        outcome_evidence_refs=(outcome,),
        revision_history=(*active.revision_history, terminal_revision),
    )
    with pytest.raises(CommitmentSystemError, match="cannot contain outcome"):
        CommitmentSystemSnapshot(records=(fulfilled,), event_receipts=snapshot.event_receipts)
    assert system.snapshot() == snapshot


def test_event_conflicts_and_time_sequence_regressions_are_atomic() -> None:
    system = CommitmentSystem()
    proposal = make_proposal()
    ingest(system, proposal, 3, event_id="event:first:ingest")
    before = system.snapshot()
    with pytest.raises(CommitmentSystemConflict, match="conflicting input"):
        system.ingest_proposal(
            proposal,
            event_for(4, proposal.evidence_refs, event_id="event:first:ingest"),
        )
    with pytest.raises(CommitmentSystemError, match="sequence regressed"):
        system.ingest_proposal(
            proposal,
            event_for(2, proposal.evidence_refs, event_id="event:older:ingest"),
        )
    with pytest.raises(CommitmentSystemError, match="time regressed"):
        system.ingest_proposal(
            proposal,
            event_for(4, proposal.evidence_refs, at=BASE_TIME + timedelta(seconds=2)),
        )
    assert system.snapshot() == before


def test_proposal_identity_binds_terms_and_rejects_conflicting_genesis() -> None:
    proposal = make_proposal()
    assert proposal.commitment_id != make_proposal(scope=("new obligation",)).commitment_id
    assert proposal.commitment_id != make_proposal(
        deadline=Deadline(BASE_TIME + timedelta(days=1))
    ).commitment_id
    system = CommitmentSystem()
    ingest(system, proposal, 1)
    before = system.snapshot()
    different_genesis = make_proposal(genesis_id="event:other:genesis")
    with pytest.raises(CommitmentSystemConflict, match="different proposal"):
        ingest(system, different_genesis, 2, event_id="event:repeat:changed-genesis")
    assert system.snapshot() == before


def test_raw_evidence_full_bound_leaves_room_for_proof_digest() -> None:
    refs = tuple(
        R13Reference(R13ReferenceKind.EVENT, f"event:bound:{index:02d}")
        for index in range(COMMITMENT_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT)
    )
    proposal = make_proposal(evidence_refs=refs)
    system, active = active_system(proposal=proposal)
    assert len(active.revision_history[-1].evidence_refs) == len(refs) + 1 == 33
    proof, event = transition_for(active, 3, CommitmentRevisionOperation.RELEASE, refs=refs)
    released = system.release(active.commitment_id, proof, event)
    assert len(released.revision_history[-1].evidence_refs) == 33

    with pytest.raises(ValueError, match="evidence_refs exceeds its bound"):
        event_for(
            4,
            (*refs, R13Reference(R13ReferenceKind.EVENT, "event:bound:overflow")),
        )
    with pytest.raises(ValueError, match="event_sequence must be a positive bounded integer"):
        CommitmentMutationEvidence(
            "event:over-sequence",
            R13_MAX_EVENT_SEQUENCE + 1,
            BASE_TIME,
            refs[:1],
        )


def test_full_u1_proposal_shape_with_maximal_references_and_scope_serializes() -> None:
    def reference(kind: R13ReferenceKind, prefix: str, index: int) -> R13Reference:
        identifier = f"{prefix}:{index:02d}"
        return R13Reference(kind, identifier + "x" * (128 - len(identifier)))

    proposal = make_proposal(
        scope=tuple(f"{index:02d}" + "s" * 254 for index in range(32)),
        beneficiary=reference(R13ReferenceKind.EXTERNAL_PARTY, "party", 0),
        deadline=Deadline(datetime.max.replace(tzinfo=UTC)),
        evidence_refs=tuple(reference(R13ReferenceKind.EVENT, "event", index)
                            for index in range(32)),
        origin_refs=tuple(reference(R13ReferenceKind.OPERATOR_PROPOSAL, "origin", index)
                          for index in range(32)),
        related_goal_refs=tuple(reference(R13ReferenceKind.GOAL, "goal", index)
                                for index in range(16)),
        desire_refs=tuple(reference(R13ReferenceKind.MOTIVATION, "motivation", index)
                          for index in range(16)),
    )
    system, active = active_system(proposal=proposal)
    assert len(active.scope) == 32
    assert len(active.evidence_refs) == 32
    assert len(active.origin_refs) == 32
    assert len(active.related_goal_refs) == 16
    assert len(active.desire_refs) == 16
    assert len(active.revision_history[-1].evidence_refs) == 33
    proof, event = transition_for(
        active, 3, CommitmentRevisionOperation.RENEGOTIATE, refs=proposal.evidence_refs
    )
    renegotiated = system.renegotiate(active.commitment_id, proof, event)
    snapshot = system.snapshot()
    assert renegotiated.lifecycle is CommitmentLifecycle.RENEGOTIATED
    assert len(renegotiated.revision_history[-1].evidence_refs) == 33
    assert snapshot.serialized_bytes <= COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES


def test_future_snapshot_and_noncanonical_mutation_evidence_fail_closed() -> None:
    system = CommitmentSystem()
    empty = system.snapshot()
    with pytest.raises(ValueError, match="unsupported CommitmentSystem snapshot version"):
        CommitmentSystemSnapshot(schema_version=2)
    with pytest.raises(ValueError):
        event_for(
            1,
            (R13Reference(R13ReferenceKind.EVENT, "event:naive"),),
            at=datetime(2026, 1, 1),
        )
    with pytest.raises(TypeError, match="evidence_refs must be a tuple"):
        CommitmentMutationEvidence(
            "event:noncanonical", 1, BASE_TIME, []  # type: ignore[arg-type]
        )
    assert system.snapshot() == empty


def test_record_receipt_and_snapshot_bytes_fail_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = CommitmentSystem()
    monkeypatch.setattr(commitment_system_module, "COMMITMENT_SYSTEM_MAX_RECORDS", 1)
    first = make_proposal()
    ingest(system, first, 1)
    before = system.snapshot()
    with pytest.raises(CommitmentSystemCapacityExceeded, match="record capacity"):
        ingest(system, make_proposal("second"), 2, event_id="event:ingest:second")
    assert system.snapshot() == before

    monkeypatch.setattr(commitment_system_module, "COMMITMENT_SYSTEM_MAX_RECORDS", 32)
    monkeypatch.setattr(commitment_system_module, "COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS", 2)
    ingest(system, first, 2, event_id="event:repeat:capacity")
    before = system.snapshot()
    with pytest.raises(CommitmentSystemCapacityExceeded, match="receipt capacity"):
        ingest(system, first, 3, event_id="event:repeat:over")
    assert system.snapshot() == before

    monkeypatch.setattr(commitment_system_module, "COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS", 256)
    monkeypatch.setattr(
        commitment_system_module, "COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES", before.serialized_bytes
    )
    with pytest.raises(CommitmentSystemCapacityExceeded, match="snapshot exceeds its byte bound"):
        ingest(system, first, 3, event_id="event:repeat:too-large")
    assert system.snapshot() == before


def test_full_record_and_receipt_bounds_accept_maximum_reject_one_over() -> None:
    assert COMMITMENT_SYSTEM_MAX_RECORDS == 32
    assert COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS == 256
    assert COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES == 16 * 1024 * 1024

    system = CommitmentSystem()
    for index in range(COMMITMENT_SYSTEM_MAX_RECORDS):
        proposal = make_proposal(f"item-{index}", genesis_id="event:batch:genesis")
        ingest(system, proposal, index + 1, event_id=None if index == 0 else f"event:ingest:{index}")
    full_records = system.snapshot()
    assert len(full_records.records) == COMMITMENT_SYSTEM_MAX_RECORDS
    with pytest.raises(CommitmentSystemCapacityExceeded, match="record capacity"):
        ingest(system, make_proposal("extra"), 33, event_id="event:ingest:overflow")
    assert system.snapshot() == full_records

    proposal = make_proposal()
    smaller_system = CommitmentSystem()
    ingest(smaller_system, proposal, 1)
    original = smaller_system.snapshot().event_receipts[0]
    receipts = (original,) + tuple(
        replace(
            original,
            event_id=f"event:repeat:{sequence}",
            event_sequence=sequence,
            recorded_at=BASE_TIME + timedelta(seconds=sequence),
        )
        for sequence in range(2, COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS + 1)
    )
    maximum = CommitmentSystemSnapshot(records=smaller_system.records, event_receipts=receipts)
    assert len(maximum.event_receipts) == COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS
    assert maximum.serialized_bytes <= COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES
    with pytest.raises(CommitmentSystemCapacityExceeded, match="receipt capacity"):
        CommitmentSystemSnapshot(
            records=smaller_system.records,
            event_receipts=(
                *receipts,
                replace(
                    original,
                    event_id="event:repeat:over",
                    event_sequence=COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS + 1,
                    recorded_at=BASE_TIME + timedelta(
                        seconds=COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS + 1
                    ),
                ),
            ),
        )


def test_no_later_authority_imports_or_production_side_effect_methods() -> None:
    source = Path(commitment_system_module.__file__).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("<relative>" if node.level else node.module or "<unknown>")
    allowed = {
        "suzka.identifiers",
        "suzka.motivation.common",
        "suzka.motivation.commitment",
    }
    assert all(
        item in allowed or item.split(".", maxsplit=1)[0] in sys.stdlib_module_names
        for item in imported
    )
    for name in (
        "fulfill", "breach", "tick", "schedule", "create_admission",
        "auto_release", "produce_proof", "persist", "restore_from_agent_state",
    ):
        assert not hasattr(CommitmentSystem, name)


def test_package_exports_include_commitment_authority() -> None:
    from suzka.motivation import CommitmentSystem as ExportedSystem
    from suzka.motivation import CommitmentSystemSnapshot as ExportedSnapshot

    assert ExportedSystem is CommitmentSystem
    assert ExportedSnapshot is CommitmentSystemSnapshot
