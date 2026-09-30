"""Focused tests for the pure R13 Goal contract boundary."""

from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import cast

import pytest

from suzka.motivation.common import (
    Deadline,
    R13Reference,
    R13ReferenceKind,
    RevisionCompactionAnchor,
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
    validate_goal_reference_graph,
)


TARGET = R13Reference(R13ReferenceKind.VALUE, "value:health")
ORIGIN = R13Reference(R13ReferenceKind.USER_REQUEST, "request:exercise")
EVIDENCE = R13Reference(R13ReferenceKind.EVENT, "event:exercise")


def make_goal(target: R13Reference = TARGET, **changes: object) -> GoalRecord:
    values: dict[str, object] = {
        "goal_id": goal_id_for_target(target),
        "target": target,
        "description": "exercise regularly",
        "origin_refs": (ORIGIN,),
        "evidence_refs": (EVIDENCE,),
    }
    values.update(changes)
    if "revision_history" not in values:
        proposal = goal_proposal_digest(
            cast(R13Reference, values["target"]),
            cast(str, values["description"]),
            deadline=cast(Deadline | datetime | None, values.get("deadline")),
            origin_refs=cast(tuple[R13Reference, ...], values["origin_refs"]),
            evidence_refs=cast(tuple[R13Reference, ...], values["evidence_refs"]),
            dependencies=cast(tuple[str, ...], values.get("dependencies", ())),
            conflicts=cast(tuple[str, ...], values.get("conflicts", ())),
        )
        values["revision_history"] = (
            GoalRevisionRecord(
                values["goal_id"],
                0,
                GoalRevisionOperation.CREATE,
                GoalRevisionReason.CREATION,
                datetime(2026, 1, 1, tzinfo=UTC),
                None,
                proposal,
                event_id="event:create",
                event_sequence=1,
                evidence_refs=(EVIDENCE.reference,),
            ),
        )
    return GoalRecord(**values)


def transition_history(
    goal: GoalRecord,
    operation: GoalRevisionOperation,
    reason: GoalRevisionReason,
    *,
    event_id: str,
    event_sequence: int = 2,
    evidence_refs: tuple[str, ...],
) -> tuple[GoalRevisionRecord, GoalRevisionRecord]:
    genesis = GoalRevisionRecord(
        goal.goal_id,
        0,
        GoalRevisionOperation.CREATE,
        GoalRevisionReason.CREATION,
        datetime(2026, 1, 1, tzinfo=UTC),
        None,
        goal.proposal_digest,
        event_id="event:create",
        event_sequence=1,
        evidence_refs=(EVIDENCE.reference,),
    )
    transition = GoalRevisionRecord(
        goal.goal_id,
        1,
        operation,
        reason,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.PROPOSED,
        goal.proposal_digest,
        event_id=event_id,
        event_sequence=event_sequence,
        evidence_refs=evidence_refs,
        previous_revision_digest=genesis.record_digest,
    )
    return genesis, transition


def admitted_terminal_history(
    goal: GoalRecord,
    admission: GoalSubjectAdmission,
    operation: GoalRevisionOperation,
    reason: GoalRevisionReason,
    *,
    event_id: str,
    event_sequence: int,
    evidence_refs: tuple[str, ...],
) -> tuple[GoalRevisionRecord, GoalRevisionRecord, GoalRevisionRecord]:
    genesis = GoalRevisionRecord(
        goal.goal_id,
        0,
        GoalRevisionOperation.CREATE,
        GoalRevisionReason.CREATION,
        datetime(2026, 1, 1, tzinfo=UTC),
        None,
        goal.proposal_digest,
        event_id="event:create",
        event_sequence=1,
        evidence_refs=(EVIDENCE.reference,),
    )
    adopted = GoalRevisionRecord(
        goal.goal_id,
        1,
        GoalRevisionOperation.ADOPT,
        GoalRevisionReason.SUBJECT_ADMISSION,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.PROPOSED,
        goal.proposal_digest,
        event_id=admission.event_id,
        event_sequence=admission.event_sequence,
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
        previous_revision_digest=genesis.record_digest,
    )
    final = GoalRevisionRecord(
        goal.goal_id,
        2,
        operation,
        reason,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.ADOPTED,
        goal.proposal_digest,
        event_id=event_id,
        event_sequence=event_sequence,
        evidence_refs=evidence_refs,
        previous_revision_digest=adopted.record_digest,
    )
    return genesis, adopted, final


def test_identity_excludes_proposal_lifecycle_proof_and_digest_binds_proposal() -> None:
    first = make_goal()
    second = make_goal(
        deadline=Deadline(datetime(2026, 2, 1, tzinfo=UTC)),
    )
    assert first.goal_id == second.goal_id
    assert first.proposal_digest != second.proposal_digest
    assert goal_proposal_digest(
        TARGET,
        "exercise regularly",
        origin_refs=(ORIGIN,),
        evidence_refs=(EVIDENCE,),
    ) == first.proposal_digest
    with pytest.raises(ValueError, match="does not bind its proposal"):
        replace(first, description="a different proposal")


def test_external_proposal_is_not_adoption_and_admission_is_exact() -> None:
    proposed = make_goal()
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "runtime:event-admit",
        2,
        GoalAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    adopted = make_goal(
        lifecycle=GoalLifecycle.ADOPTED,
        subject_admission=admission,
        revision=1,
        revision_history=transition_history(
            proposed,
            GoalRevisionOperation.ADOPT,
            GoalRevisionReason.SUBJECT_ADMISSION,
            event_id=admission.event_id,
            event_sequence=admission.event_sequence,
            evidence_refs=tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            ),
        ),
    )
    assert proposed.lifecycle is GoalLifecycle.PROPOSED
    assert adopted.lifecycle is GoalLifecycle.ADOPTED
    assert adopted.admission is admission
    assert admission.digest
    with pytest.raises(ValueError):
        make_goal(lifecycle=GoalLifecycle.PROPOSED, subject_admission=admission)
    changed_admission = replace(admission, event_sequence=3)
    assert changed_admission.digest != admission.digest


@pytest.mark.parametrize(
    ("operation", "reason", "next_lifecycle"),
    [
        (
            GoalRevisionOperation.DEFER,
            GoalRevisionReason.SUBJECT_DEFERRED,
            GoalLifecycle.DEFERRED,
        ),
        (
            GoalRevisionOperation.ABANDON,
            GoalRevisionReason.SUBJECT_ABANDONED,
            GoalLifecycle.ABANDONED,
        ),
    ],
)
def test_post_admission_goal_decisions_require_fresh_transition_proofs(
    operation: GoalRevisionOperation,
    reason: GoalRevisionReason,
    next_lifecycle: GoalLifecycle,
) -> None:
    proposed = make_goal()
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "event:admit-before-transition",
        2,
        GoalAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    adopted = make_goal(
        lifecycle=GoalLifecycle.ADOPTED,
        subject_admission=admission,
        revision=1,
        revision_history=transition_history(
            proposed,
            GoalRevisionOperation.ADOPT,
            GoalRevisionReason.SUBJECT_ADMISSION,
            event_id=admission.event_id,
            event_sequence=admission.event_sequence,
            evidence_refs=tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            ),
        ),
    )
    stale = GoalRevisionRecord(
        adopted.goal_id,
        2,
        operation,
        reason,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.ADOPTED,
        adopted.proposal_digest,
        event_id="event:later-subject-decision",
        event_sequence=3,
        evidence_refs=(admission.admission_digest,),
        previous_revision_digest=adopted.revision_history[-1].record_digest,
    )
    with pytest.raises(ValueError, match="fresh subject proof"):
        replace(
            adopted,
            lifecycle=next_lifecycle,
            revision=2,
            revision_history=(*adopted.revision_history, stale),
        )

    proof = GoalSubjectTransitionProof(
        adopted.goal_id,
        adopted.proposal_digest,
        operation,
        reason,
        GoalLifecycle.ADOPTED,
        (f"request:decision:{operation.value}",),
        "event:later-subject-decision",
        3,
    )
    revision = replace(
        stale,
        evidence_refs=tuple(sorted((*proof.evidence_refs, proof.transition_digest))),
    )
    decided = replace(
        adopted,
        lifecycle=next_lifecycle,
        revision=2,
        revision_history=(*adopted.revision_history, revision),
        subject_transition_proofs=(proof,),
    )

    assert decided.lifecycle is next_lifecycle


def test_creation_and_adoption_may_share_their_exact_event_witness() -> None:
    proposed = make_goal()
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "event:create-and-admit",
        1,
        GoalAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    genesis = GoalRevisionRecord(
        proposed.goal_id,
        0,
        GoalRevisionOperation.CREATE,
        GoalRevisionReason.CREATION,
        datetime(2026, 1, 1, tzinfo=UTC),
        None,
        proposed.proposal_digest,
        event_id=admission.event_id,
        event_sequence=admission.event_sequence,
        evidence_refs=(EVIDENCE.reference,),
    )
    adopted_revision = GoalRevisionRecord(
        proposed.goal_id,
        1,
        GoalRevisionOperation.ADOPT,
        GoalRevisionReason.SUBJECT_ADMISSION,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.PROPOSED,
        proposed.proposal_digest,
        event_id=admission.event_id,
        event_sequence=admission.event_sequence,
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
        previous_revision_digest=genesis.record_digest,
    )

    adopted = make_goal(
        lifecycle=GoalLifecycle.ADOPTED,
        subject_admission=admission,
        revision=1,
        revision_history=(genesis, adopted_revision),
    )

    assert adopted.lifecycle is GoalLifecycle.ADOPTED


def test_deferred_goal_requires_fresh_event_bound_proof_to_reactivate() -> None:
    proposed = make_goal()
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "event:defer",
        2,
        GoalAdmissionReason.SUBJECT_REVIEW,
    )
    deferred = make_goal(
        lifecycle=GoalLifecycle.DEFERRED,
        subject_admission=admission,
        revision=1,
        revision_history=transition_history(
            proposed,
            GoalRevisionOperation.DEFER,
            GoalRevisionReason.SUBJECT_DEFERRED,
            event_id=admission.event_id,
            event_sequence=admission.event_sequence,
            evidence_refs=tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            ),
        ),
    )
    reactivation = GoalRevisionRecord(
        deferred.goal_id,
        2,
        GoalRevisionOperation.ADOPT,
        GoalRevisionReason.SUBJECT_ADMISSION,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.DEFERRED,
        deferred.proposal_digest,
        event_id="event:reactivate",
        event_sequence=3,
        evidence_refs=(admission.admission_digest,),
        previous_revision_digest=deferred.revision_history[-1].record_digest,
    )

    with pytest.raises(ValueError, match="fresh subject proof"):
        replace(
            deferred,
            lifecycle=GoalLifecycle.ADOPTED,
            revision=2,
            revision_history=(*deferred.revision_history, reactivation),
        )

    proof = GoalSubjectTransitionProof(
        deferred.goal_id,
        deferred.proposal_digest,
        GoalRevisionOperation.ADOPT,
        GoalRevisionReason.SUBJECT_ADMISSION,
        GoalLifecycle.DEFERRED,
        ("request:re-adopt",),
        "event:reactivate",
        3,
    )
    reactivation = replace(
        reactivation,
        evidence_refs=tuple(sorted((*proof.evidence_refs, proof.transition_digest))),
    )
    reactivated = replace(
        deferred,
        lifecycle=GoalLifecycle.ADOPTED,
        revision=2,
        revision_history=(*deferred.revision_history, reactivation),
        subject_transition_proofs=(proof,),
    )
    assert reactivated.lifecycle is GoalLifecycle.ADOPTED

    mismatched_proof = replace(proof, event_sequence=4)
    with pytest.raises(ValueError, match="fresh subject proof"):
        replace(
            deferred,
            lifecycle=GoalLifecycle.ADOPTED,
            revision=2,
            revision_history=(*deferred.revision_history, reactivation),
            subject_transition_proofs=(mismatched_proof,),
        )


def test_goal_compaction_anchor_recomputes_its_anchored_revision_digest() -> None:
    proposed = make_goal()
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "event:defer:1",
        2,
        GoalAdmissionReason.SUBJECT_REVIEW,
    )
    genesis = GoalRevisionRecord(
        proposed.goal_id,
        0,
        GoalRevisionOperation.CREATE,
        GoalRevisionReason.CREATION,
        datetime(2026, 1, 1, tzinfo=UTC),
        None,
        proposed.proposal_digest,
        event_id="event:create",
        event_sequence=1,
        evidence_refs=(EVIDENCE.reference,),
    )
    revisions = [genesis]
    transition_proofs = []
    previous = genesis.record_digest
    for number in range(1, 9):
        event_id = admission.event_id if number == 1 else f"event:defer:{number}"
        event_sequence = admission.event_sequence if number == 1 else number + 1
        if number == 1:
            decision_evidence = tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            )
        else:
            proof = GoalSubjectTransitionProof(
                proposed.goal_id,
                proposed.proposal_digest,
                GoalRevisionOperation.DEFER,
                GoalRevisionReason.SUBJECT_DEFERRED,
                GoalLifecycle.DEFERRED,
                (f"request:defer:{number}",),
                event_id,
                event_sequence,
            )
            transition_proofs.append(proof)
            decision_evidence = tuple(
                sorted((*proof.evidence_refs, proof.transition_digest))
            )
        revision = GoalRevisionRecord(
            proposed.goal_id,
            number,
            GoalRevisionOperation.DEFER,
            GoalRevisionReason.SUBJECT_DEFERRED,
            datetime(2026, 1, 1, tzinfo=UTC),
            GoalLifecycle.PROPOSED
            if number == 1
            else GoalLifecycle.DEFERRED,
            proposed.proposal_digest,
            event_id=event_id,
            event_sequence=event_sequence,
            evidence_refs=decision_evidence,
            previous_revision_digest=previous,
        )
        revisions.append(revision)
        previous = revision.record_digest
    anchor = RevisionCompactionAnchor(
        authority_id=proposed.goal_id,
        through_revision=0,
        through_digest=genesis.record_digest,
        through_created_at=genesis.created_at,
        through_evidence_refs=genesis.evidence_refs,
        through_previous_revision_digest=None,
        through_state=GoalLifecycle.PROPOSED.value,
        through_previous_state=None,
        through_operation=GoalRevisionOperation.CREATE.value,
        through_reason=GoalRevisionReason.CREATION.value,
        through_event_id="event:create",
        through_event_sequence=1,
        through_proposal_digest=proposed.proposal_digest,
    )
    deferred = make_goal(
        lifecycle=GoalLifecycle.DEFERRED,
        subject_admission=admission,
        revision=8,
        revision_history=tuple(revisions[1:]),
        history_anchor=anchor,
        subject_transition_proofs=tuple(
            sorted(transition_proofs, key=lambda proof: proof.transition_digest)
        ),
    )

    assert deferred.lifecycle is GoalLifecycle.DEFERRED
    with pytest.raises(ValueError, match="anchor does not match"):
        replace(
            deferred,
            history_anchor=replace(anchor, through_event_id="event:forged"),
        )


def test_compacted_goal_anchor_must_retain_its_initial_admission_proof() -> None:
    proposed = make_goal()
    genesis = proposed.revision_history[0]
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "event:actual-admission",
        2,
        GoalAdmissionReason.SUBJECT_REVIEW,
    )
    invalid_compacted_revision = GoalRevisionRecord(
        proposed.goal_id,
        1,
        GoalRevisionOperation.DEFER,
        GoalRevisionReason.SUBJECT_DEFERRED,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.PROPOSED,
        proposed.proposal_digest,
        event_id="event:missing-admission-proof",
        event_sequence=2,
        evidence_refs=("event:unrelated",),
        previous_revision_digest=genesis.record_digest,
    )
    retained = []
    previous_digest = invalid_compacted_revision.record_digest
    for number in range(2, 10):
        revision = GoalRevisionRecord(
            proposed.goal_id,
            number,
            GoalRevisionOperation.DEFER,
            GoalRevisionReason.SUBJECT_DEFERRED,
            datetime(2026, 1, 1, tzinfo=UTC),
            GoalLifecycle.DEFERRED,
            proposed.proposal_digest,
            event_id=f"event:retained-defer:{number}",
            event_sequence=number + 1,
            evidence_refs=(admission.admission_digest,),
            previous_revision_digest=previous_digest,
        )
        retained.append(revision)
        previous_digest = revision.record_digest
    anchor = RevisionCompactionAnchor(
        authority_id=proposed.goal_id,
        through_revision=1,
        through_digest=invalid_compacted_revision.record_digest,
        through_created_at=invalid_compacted_revision.created_at,
        through_evidence_refs=invalid_compacted_revision.evidence_refs,
        through_previous_revision_digest=genesis.record_digest,
        through_state=GoalLifecycle.DEFERRED.value,
        through_previous_state=GoalLifecycle.PROPOSED.value,
        through_operation=GoalRevisionOperation.DEFER.value,
        through_reason=GoalRevisionReason.SUBJECT_DEFERRED.value,
        through_event_id="event:missing-admission-proof",
        through_event_sequence=2,
        through_proposal_digest=proposed.proposal_digest,
    )

    with pytest.raises(ValueError, match="exact subject admission"):
        make_goal(
            lifecycle=GoalLifecycle.DEFERRED,
            subject_admission=admission,
            revision=9,
            revision_history=tuple(retained),
            history_anchor=anchor,
        )


def test_compacted_post_admission_goal_transition_retains_fresh_proof() -> None:
    proposed = make_goal()
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "event:initial-defer",
        2,
        GoalAdmissionReason.SUBJECT_REVIEW,
    )
    genesis = proposed.revision_history[0]
    initial_defer = GoalRevisionRecord(
        proposed.goal_id,
        1,
        GoalRevisionOperation.DEFER,
        GoalRevisionReason.SUBJECT_DEFERRED,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.PROPOSED,
        proposed.proposal_digest,
        event_id=admission.event_id,
        event_sequence=admission.event_sequence,
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
        previous_revision_digest=genesis.record_digest,
    )
    anchor_proof = GoalSubjectTransitionProof(
        proposed.goal_id,
        proposed.proposal_digest,
        GoalRevisionOperation.DEFER,
        GoalRevisionReason.SUBJECT_DEFERRED,
        GoalLifecycle.DEFERRED,
        ("request:defer:2",),
        "event:defer:2",
        3,
    )
    anchor_revision = GoalRevisionRecord(
        proposed.goal_id,
        2,
        GoalRevisionOperation.DEFER,
        GoalRevisionReason.SUBJECT_DEFERRED,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.DEFERRED,
        proposed.proposal_digest,
        event_id=anchor_proof.event_id,
        event_sequence=anchor_proof.event_sequence,
        evidence_refs=tuple(
            sorted((*anchor_proof.evidence_refs, anchor_proof.transition_digest))
        ),
        previous_revision_digest=initial_defer.record_digest,
    )
    proofs = [anchor_proof]
    retained = []
    previous_digest = anchor_revision.record_digest
    for number in range(3, 11):
        proof = GoalSubjectTransitionProof(
            proposed.goal_id,
            proposed.proposal_digest,
            GoalRevisionOperation.DEFER,
            GoalRevisionReason.SUBJECT_DEFERRED,
            GoalLifecycle.DEFERRED,
            (f"request:defer:{number}",),
            f"event:defer:{number}",
            number + 1,
        )
        revision = GoalRevisionRecord(
            proposed.goal_id,
            number,
            GoalRevisionOperation.DEFER,
            GoalRevisionReason.SUBJECT_DEFERRED,
            datetime(2026, 1, 1, tzinfo=UTC),
            GoalLifecycle.DEFERRED,
            proposed.proposal_digest,
            event_id=proof.event_id,
            event_sequence=proof.event_sequence,
            evidence_refs=tuple(sorted((*proof.evidence_refs, proof.transition_digest))),
            previous_revision_digest=previous_digest,
        )
        retained.append(revision)
        proofs.append(proof)
        previous_digest = revision.record_digest

    anchor = RevisionCompactionAnchor(
        authority_id=proposed.goal_id,
        through_revision=2,
        through_digest=anchor_revision.record_digest,
        through_created_at=anchor_revision.created_at,
        through_evidence_refs=anchor_revision.evidence_refs,
        through_previous_revision_digest=initial_defer.record_digest,
        through_state=GoalLifecycle.DEFERRED.value,
        through_previous_state=GoalLifecycle.DEFERRED.value,
        through_operation=GoalRevisionOperation.DEFER.value,
        through_reason=GoalRevisionReason.SUBJECT_DEFERRED.value,
        through_event_id=anchor_revision.event_id,
        through_event_sequence=anchor_revision.event_sequence,
        through_proposal_digest=proposed.proposal_digest,
    )
    compacted = make_goal(
        lifecycle=GoalLifecycle.DEFERRED,
        subject_admission=admission,
        subject_transition_proofs=tuple(
            sorted(proofs, key=lambda proof: proof.transition_digest)
        ),
        revision=10,
        revision_history=tuple(retained),
        history_anchor=anchor,
    )

    assert compacted.lifecycle is GoalLifecycle.DEFERRED


def test_subject_abandonment_is_not_verified_failure() -> None:
    proposed = make_goal()
    admission = GoalSubjectAdmission(
        proposed.goal_id,
        proposed.proposal_digest,
        (EVIDENCE.reference,),
        "runtime:event-abandon",
        2,
        GoalAdmissionReason.SUBJECT_REVIEW,
    )
    abandoned = make_goal(
        lifecycle=GoalLifecycle.ABANDONED,
        subject_admission=admission,
        revision=1,
        revision_history=transition_history(
            proposed,
            GoalRevisionOperation.ABANDON,
            GoalRevisionReason.SUBJECT_ABANDONED,
            event_id=admission.event_id,
            event_sequence=admission.event_sequence,
            evidence_refs=tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            ),
        ),
    )
    assert not abandoned.outcome_evidence_refs
    reopen_revision = GoalRevisionRecord(
        abandoned.goal_id,
        2,
        GoalRevisionOperation.ADOPT,
        GoalRevisionReason.SUBJECT_ADMISSION,
        datetime(2026, 1, 1, tzinfo=UTC),
        GoalLifecycle.ABANDONED,
        abandoned.proposal_digest,
        event_id="runtime:event-reopen",
        event_sequence=3,
        evidence_refs=(admission.admission_digest,),
        previous_revision_digest=abandoned.revision_history[-1].record_digest,
    )
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        replace(
            abandoned,
            lifecycle=GoalLifecycle.ADOPTED,
            revision=2,
            revision_history=(*abandoned.revision_history, reopen_revision),
        )
    with pytest.raises(ValueError):
        make_goal(
            lifecycle=GoalLifecycle.FAILED,
            subject_admission=admission,
            revision=2,
            revision_history=admitted_terminal_history(
                proposed,
                admission,
                GoalRevisionOperation.FAIL,
                GoalRevisionReason.VERIFIED_OUTCOME,
                event_id="runtime:event-fail",
                event_sequence=3,
                evidence_refs=("event:outcome-missing",),
            ),
        )
    outcome = R13Reference(R13ReferenceKind.EVENT, "event:verified")
    failed = make_goal(
        lifecycle=GoalLifecycle.FAILED,
        subject_admission=admission,
        outcome_evidence_refs=(outcome,),
        revision=2,
        revision_history=admitted_terminal_history(
            proposed,
            admission,
            GoalRevisionOperation.FAIL,
            GoalRevisionReason.VERIFIED_OUTCOME,
            event_id="runtime:event-fail",
            event_sequence=3,
            evidence_refs=(outcome.reference,),
        ),
    )
    assert failed.lifecycle is GoalLifecycle.FAILED


def test_dependency_and_conflict_graphs_fail_closed_without_winner_selection() -> None:
    target_a = R13Reference(R13ReferenceKind.STATE, "state:a")
    target_b = R13Reference(R13ReferenceKind.STATE, "state:b")
    goal_a_id = goal_id_for_target(target_a)
    goal_b_id = goal_id_for_target(target_b)
    goal_a = make_goal(target_a, dependencies=(goal_b_id,))
    goal_b = make_goal(target_b, dependencies=(goal_a_id,))
    with pytest.raises(ValueError, match="cycle"):
        validate_goal_reference_graph((goal_a, goal_b))
    with pytest.raises(ValueError, match="missing"):
        validate_goal_reference_graph((make_goal(dependencies=("missing",)),))
    with pytest.raises(ValueError):
        make_goal(dependencies=(goal_id_for_target(TARGET),))


def test_bounds_closed_vocabulary_and_private_surface() -> None:
    with pytest.raises(ValueError):
        make_goal(description="x" * 1_025)
    with pytest.raises(ValueError):
        make_goal(evidence_refs=(EVIDENCE, EVIDENCE))
    self_evidence = R13Reference(
        R13ReferenceKind.GOAL,
        goal_id_for_target(TARGET),
    )
    with pytest.raises(ValueError, match="cannot use itself as evidence"):
        make_goal(evidence_refs=(self_evidence,))
    names = {item.name.casefold() for item in fields(GoalRecord)}
    assert not names.intersection({"prompt", "transcript", "hidden_thought", "rationale"})
    with pytest.raises(TypeError):
        make_goal(metadata={"prompt": "private"})
