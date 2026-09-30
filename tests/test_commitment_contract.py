"""Focused tests for the pure R13 Commitment contract boundary."""

from dataclasses import fields, replace
from datetime import UTC, datetime

import pytest

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
    commitment_proposal_digest,
)
from suzka.motivation.common import Deadline, R13Reference, R13ReferenceKind


BENEFICIARY = R13Reference(R13ReferenceKind.SUBJECT, "subject:self")
ORIGIN = R13Reference(R13ReferenceKind.OPERATOR_PROPOSAL, "proposal:operator")
EVIDENCE = R13Reference(R13ReferenceKind.EVENT, "event:proposal")


def make_commitment(**changes: object) -> CommitmentRecord:
    values: dict[str, object] = {
        "subject": "write the report",
        "beneficiary": BENEFICIARY,
        "scope": ("report",),
        "evidence_refs": (EVIDENCE,),
        "origin_refs": (ORIGIN,),
    }
    values.update(changes)
    if "revision_history" not in values:
        commitment_id = commitment_id_for_fields(
            subject=values["subject"],
            beneficiary=values["beneficiary"],
            scope=values["scope"],
            deadline=values.get("deadline"),
            evidence_refs=values["evidence_refs"],
            origin_refs=values["origin_refs"],
            related_goal_refs=values.get("related_goal_refs", ()),
            desire_refs=values.get("desire_refs", ()),
        )
        values["revision_history"] = (
            CommitmentRevisionRecord(
                commitment_id,
                0,
                CommitmentRevisionOperation.CREATE,
                CommitmentRevisionReason.CREATION,
                datetime(2026, 1, 1, tzinfo=UTC),
                None,
                event_id="event:create",
                event_sequence=1,
                evidence_refs=(EVIDENCE.reference,),
            ),
        )
    return CommitmentRecord(**values)


def admission_for(record: CommitmentRecord) -> CommitmentSubjectAdmission:
    return CommitmentSubjectAdmission(
        record.commitment_id,
        record.proposal_digest,
        record.beneficiary,
        record.scope,
        record.deadline,
        tuple(item.reference for item in record.evidence_refs),
        "runtime:event-accept",
        2,
        CommitmentAdmissionReason.SUBJECT_RESPONSIBILITY,
    )


def transition_history(
    record: CommitmentRecord,
    operation: CommitmentRevisionOperation,
    reason: CommitmentRevisionReason,
    *,
    event_id: str,
    evidence_refs: tuple[str, ...],
) -> tuple[CommitmentRevisionRecord, CommitmentRevisionRecord]:
    genesis = CommitmentRevisionRecord(
        record.commitment_id,
        0,
        CommitmentRevisionOperation.CREATE,
        CommitmentRevisionReason.CREATION,
        datetime(2026, 1, 1, tzinfo=UTC),
        None,
        event_id="event:create",
        event_sequence=1,
        evidence_refs=(EVIDENCE.reference,),
    )
    transition = CommitmentRevisionRecord(
        record.commitment_id,
        1,
        operation,
        reason,
        datetime(2026, 1, 1, tzinfo=UTC),
        CommitmentLifecycle.PROPOSED,
        event_id=event_id,
        event_sequence=2,
        evidence_refs=evidence_refs,
        previous_revision_digest=genesis.record_digest,
    )
    return genesis, transition


def admitted_terminal_history(
    record: CommitmentRecord,
    admission: CommitmentSubjectAdmission,
    operation: CommitmentRevisionOperation,
    reason: CommitmentRevisionReason,
    *,
    event_id: str,
    evidence_refs: tuple[str, ...],
) -> tuple[
    CommitmentRevisionRecord,
    CommitmentRevisionRecord,
    CommitmentRevisionRecord,
]:
    genesis = CommitmentRevisionRecord(
        record.commitment_id,
        0,
        CommitmentRevisionOperation.CREATE,
        CommitmentRevisionReason.CREATION,
        datetime(2026, 1, 1, tzinfo=UTC),
        None,
        event_id="event:create",
        event_sequence=1,
        evidence_refs=(EVIDENCE.reference,),
    )
    admitted = CommitmentRevisionRecord(
        record.commitment_id,
        1,
        CommitmentRevisionOperation.ADMIT,
        CommitmentRevisionReason.SUBJECT_ADMISSION,
        datetime(2026, 1, 1, tzinfo=UTC),
        CommitmentLifecycle.PROPOSED,
        event_id=admission.event_id,
        event_sequence=admission.event_sequence,
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
        previous_revision_digest=genesis.record_digest,
    )
    final = CommitmentRevisionRecord(
        record.commitment_id,
        2,
        operation,
        reason,
        datetime(2026, 1, 1, tzinfo=UTC),
        CommitmentLifecycle.ACTIVE,
        event_id=event_id,
        event_sequence=3,
        evidence_refs=evidence_refs,
        previous_revision_digest=admitted.record_digest,
    )
    return genesis, admitted, final


def test_proposal_and_active_responsibility_are_distinct() -> None:
    proposed = make_commitment()
    admission = admission_for(proposed)
    active = make_commitment(
        lifecycle=CommitmentLifecycle.ACTIVE,
        subject_admission=admission,
        revision=1,
        revision_history=transition_history(
            proposed,
            CommitmentRevisionOperation.ADMIT,
            CommitmentRevisionReason.SUBJECT_ADMISSION,
            event_id=admission.event_id,
            evidence_refs=tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            ),
        ),
    )
    assert proposed.commitment_id == active.commitment_id
    assert proposed.lifecycle is CommitmentLifecycle.PROPOSED
    assert active.lifecycle is CommitmentLifecycle.ACTIVE
    assert active.admission is admission


def test_admission_digest_binds_scope_beneficiary_deadline_evidence_and_event() -> None:
    proposed = make_commitment(
        deadline=Deadline(datetime(2026, 2, 1, tzinfo=UTC)),
    )
    admission = admission_for(proposed)
    assert admission.digest
    changed_event = CommitmentSubjectAdmission(
        proposed.commitment_id,
        proposed.proposal_digest,
        proposed.beneficiary,
        proposed.scope,
        proposed.deadline,
        (EVIDENCE.reference,),
        "runtime:event-other",
        2,
        CommitmentAdmissionReason.SUBJECT_RESPONSIBILITY,
    )
    assert changed_event.digest != admission.digest
    assert commitment_proposal_digest(
        subject=proposed.subject,
        beneficiary=proposed.beneficiary,
        scope=proposed.scope,
        deadline=proposed.deadline,
        evidence_refs=proposed.evidence_refs,
        origin_refs=proposed.origin_refs,
    ) == proposed.proposal_digest


def test_operator_proposal_is_not_subject_admission_and_release_is_not_fulfillment() -> None:
    proposed = make_commitment()
    assert proposed.lifecycle is CommitmentLifecycle.PROPOSED
    with pytest.raises(ValueError):
        make_commitment(lifecycle=CommitmentLifecycle.RELEASED)
    admission = admission_for(proposed)
    release_proof = CommitmentSubjectTransitionProof(
        proposed.commitment_id,
        proposed.proposal_digest,
        proposed.beneficiary,
        proposed.scope,
        proposed.deadline,
        CommitmentRevisionOperation.RELEASE,
        CommitmentRevisionReason.SUBJECT_RELEASE,
        CommitmentLifecycle.ACTIVE,
        ("event:release",),
        "runtime:event-release",
        3,
    )
    release_history = admitted_terminal_history(
        proposed,
        admission,
        CommitmentRevisionOperation.RELEASE,
        CommitmentRevisionReason.SUBJECT_RELEASE,
        event_id=release_proof.event_id,
        evidence_refs=tuple(
            sorted((*release_proof.evidence_refs, release_proof.transition_digest))
        ),
    )
    stale_admission_history = admitted_terminal_history(
        proposed,
        admission,
        CommitmentRevisionOperation.RELEASE,
        CommitmentRevisionReason.SUBJECT_RELEASE,
        event_id="runtime:event-release",
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
    )
    with pytest.raises(ValueError, match="fresh subject proof"):
        make_commitment(
            lifecycle=CommitmentLifecycle.RELEASED,
            subject_admission=admission,
            revision=2,
            revision_history=stale_admission_history,
        )
    with pytest.raises(ValueError, match="fresh subject proof"):
        make_commitment(
            lifecycle=CommitmentLifecycle.RELEASED,
            subject_admission=admission,
            revision=2,
            revision_history=release_history,
        )
    released = make_commitment(
        lifecycle=CommitmentLifecycle.RELEASED,
        subject_admission=admission,
        subject_transition_proofs=(release_proof,),
        revision=2,
        revision_history=release_history,
    )
    assert released.lifecycle is CommitmentLifecycle.RELEASED
    reactivation = CommitmentRevisionRecord(
        released.commitment_id,
        3,
        CommitmentRevisionOperation.ADMIT,
        CommitmentRevisionReason.SUBJECT_ADMISSION,
        datetime(2026, 1, 1, tzinfo=UTC),
        CommitmentLifecycle.RELEASED,
        event_id="runtime:event-reactivate",
        event_sequence=4,
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
        previous_revision_digest=released.revision_history[-1].record_digest,
    )
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        replace(
            released,
            lifecycle=CommitmentLifecycle.ACTIVE,
            revision=3,
            revision_history=(*released.revision_history, reactivation),
        )
    with pytest.raises(ValueError):
        make_commitment(
            lifecycle=CommitmentLifecycle.FULFILLED,
            subject_admission=admission,
            revision=2,
            revision_history=admitted_terminal_history(
                proposed,
                admission,
                CommitmentRevisionOperation.FULFILL,
                CommitmentRevisionReason.VERIFIED_OUTCOME,
                event_id="runtime:event-fulfill",
                evidence_refs=("event:fulfill",),
            ),
        )
    outcome = R13Reference(R13ReferenceKind.EVENT, "event:verified")
    fulfilled = make_commitment(
        lifecycle=CommitmentLifecycle.FULFILLED,
        subject_admission=admission,
        outcome_evidence_refs=(outcome,),
        revision=2,
        revision_history=admitted_terminal_history(
            proposed,
            admission,
            CommitmentRevisionOperation.FULFILL,
            CommitmentRevisionReason.VERIFIED_OUTCOME,
            event_id="runtime:event-fulfill",
            evidence_refs=(outcome.reference,),
        ),
    )
    assert fulfilled.lifecycle is CommitmentLifecycle.FULFILLED


def test_commitment_renegotiation_requires_its_own_subject_proof() -> None:
    proposed = make_commitment()
    admission = admission_for(proposed)
    proof = CommitmentSubjectTransitionProof(
        proposed.commitment_id,
        proposed.proposal_digest,
        proposed.beneficiary,
        proposed.scope,
        proposed.deadline,
        CommitmentRevisionOperation.RENEGOTIATE,
        CommitmentRevisionReason.SUBJECT_RENEGOTIATION,
        CommitmentLifecycle.ACTIVE,
        ("event:renegotiate",),
        "runtime:event-renegotiate",
        3,
    )
    history = admitted_terminal_history(
        proposed,
        admission,
        CommitmentRevisionOperation.RENEGOTIATE,
        CommitmentRevisionReason.SUBJECT_RENEGOTIATION,
        event_id=proof.event_id,
        evidence_refs=tuple(
            sorted((*proof.evidence_refs, proof.transition_digest))
        ),
    )
    stale_admission_history = admitted_terminal_history(
        proposed,
        admission,
        CommitmentRevisionOperation.RENEGOTIATE,
        CommitmentRevisionReason.SUBJECT_RENEGOTIATION,
        event_id="runtime:event-renegotiate",
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
    )
    with pytest.raises(ValueError, match="fresh subject proof"):
        make_commitment(
            lifecycle=CommitmentLifecycle.RENEGOTIATED,
            subject_admission=admission,
            revision=2,
            revision_history=stale_admission_history,
        )
    with pytest.raises(ValueError, match="fresh subject proof"):
        make_commitment(
            lifecycle=CommitmentLifecycle.RENEGOTIATED,
            subject_admission=admission,
            revision=2,
            revision_history=history,
        )
    renegotiated = make_commitment(
        lifecycle=CommitmentLifecycle.RENEGOTIATED,
        subject_admission=admission,
        subject_transition_proofs=(proof,),
        revision=2,
        revision_history=history,
    )

    assert renegotiated.lifecycle is CommitmentLifecycle.RENEGOTIATED


def test_desire_is_only_a_reference_and_scope_bounds_are_explicit() -> None:
    desire = R13Reference(R13ReferenceKind.MOTIVATION, "motivation:desire")
    record = make_commitment(desire_refs=(desire,))
    assert record.desire_refs == (desire,)
    with pytest.raises(ValueError):
        make_commitment(scope=tuple(f"scope:{index:02d}" for index in range(33)))
    with pytest.raises(ValueError):
        make_commitment(scope=("x" * 257,))


def test_private_payload_fields_are_not_available() -> None:
    names = {item.name.casefold() for item in fields(CommitmentRecord)}
    assert not names.intersection({"prompt", "transcript", "hidden_thought", "rationale"})
    with pytest.raises(TypeError):
        make_commitment(metadata={"prompt": "private"})
