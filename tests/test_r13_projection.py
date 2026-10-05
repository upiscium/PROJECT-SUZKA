"""Contracts for the bounded, non-authoritative R13 prompt projection."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import json
import subprocess
import sys

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
    commitment_id_for_proposal,
    commitment_proposal_digest,
)
from suzka.motivation.commitment_system import (
    CommitmentMutationEvidence,
    CommitmentSystem,
    CommitmentSystemSnapshot,
)
from suzka.motivation.common import (
    Deadline,
    R13Reference,
    R13ReferenceKind,
    canonical_json,
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
    GoalMutationEvidence,
    GoalSystem,
    GoalSystemSnapshot,
)
from suzka.motivation.motivation import (
    MotivationKind,
    MotivationLifecycle,
)
from suzka.motivation.projection import (
    CommitmentPromptEntry,
    GoalPromptEntry,
    MotivationPromptEntry,
    R13_PROMPT_MAX_RECORDS_PER_DOMAIN,
    R13_PROMPT_MAX_RENDERED_BYTES,
    R13_PROMPT_MAX_SERIALIZED_BYTES,
    R13_PROMPT_SCHEMA_VERSION,
    R13PromptView,
)
from suzka.motivation.system import (
    MotivationEvidence,
    MotivationMutationEvidence,
    MotivationSystem,
)


BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _at(sequence: int) -> datetime:
    return BASE_TIME + timedelta(seconds=sequence)


def _goal_event(
    sequence: int,
    evidence_refs: tuple[R13Reference, ...],
    *,
    event_id: str,
) -> GoalMutationEvidence:
    return GoalMutationEvidence(event_id, sequence, _at(sequence), evidence_refs)


def _commitment_event(
    sequence: int,
    evidence_refs: tuple[R13Reference, ...],
    *,
    event_id: str,
) -> CommitmentMutationEvidence:
    return CommitmentMutationEvidence(event_id, sequence, _at(sequence), evidence_refs)


def _make_motivation_system(
    lifecycles: tuple[MotivationLifecycle, ...],
    *,
    salience: float = 1.0,
    confidence: float = 1.0,
    persistence: float = 0.5,
    uncertainty: float = 0.0,
) -> MotivationSystem:
    system = MotivationSystem()
    records = []
    for index, lifecycle in enumerate(lifecycles):
        source = R13Reference(R13ReferenceKind.EXPERIENCE, f"experience:source:{index:03}")
        evidence = MotivationEvidence(
            source_ref=source,
            target=R13Reference(R13ReferenceKind.STATE, f"state:target:{index:03}"),
            kind=MotivationKind.DESIRE,
            salience=salience,
            confidence=confidence,
            persistence=persistence,
            uncertainty=uncertainty,
            origin_refs=(
                R13Reference(R13ReferenceKind.USER_REQUEST, f"request:origin:{index:03}"),
            ),
        )
        sequence = index + 1
        records.append(
            system.apply_evidence(
                evidence,
                MotivationMutationEvidence(
                    f"event:motivation:create:{index:03}",
                    sequence,
                    _at(sequence),
                    (source,),
                ),
            )
        )

    next_sequence = len(lifecycles) + 1
    for index, lifecycle in enumerate(lifecycles):
        record = records[index]
        if lifecycle is MotivationLifecycle.ACTIVE:
            continue
        event = MotivationMutationEvidence(
            f"event:motivation:transition:{index:03}",
            next_sequence,
            _at(next_sequence),
            (R13Reference(R13ReferenceKind.MOTIVATION, record.motivation_id),),
        )
        if lifecycle is MotivationLifecycle.DORMANT:
            system.decay(event, elapsed_seconds=86_400.0)
        elif lifecycle is MotivationLifecycle.SATIATED:
            system.satiate(record.motivation_id, event)
        elif lifecycle is MotivationLifecycle.RETIRED:
            system.retire(record.motivation_id, event)
        else:
            raise AssertionError(f"unhandled Motivation lifecycle: {lifecycle}")
        next_sequence += 1
    return system


def _goal_proposal(
    index: int,
    *,
    description: str | None = None,
    deadline: Deadline | None = None,
) -> GoalRecord:
    target = R13Reference(R13ReferenceKind.VALUE, f"value:goal:{index:03}")
    origin = R13Reference(R13ReferenceKind.USER_REQUEST, f"request:goal:{index:03}")
    evidence = (R13Reference(R13ReferenceKind.EVENT, f"event:goal:evidence:{index:03}"),)
    text = description or f"goal description {index:03}"
    actual_deadline = deadline if deadline is not None else Deadline()
    proposal_digest = goal_proposal_digest(
        target,
        text,
        deadline=actual_deadline,
        origin_refs=(origin,),
        evidence_refs=evidence,
    )
    goal_id = goal_id_for_target(target)
    sequence = index + 1
    genesis = GoalRevisionRecord(
        goal_id=goal_id,
        revision=0,
        operation=GoalRevisionOperation.CREATE,
        reason=GoalRevisionReason.CREATION,
        created_at=_at(sequence),
        previous_lifecycle_state=None,
        proposal_digest=proposal_digest,
        event_id=f"event:goal:create:{index:03}",
        event_sequence=sequence,
        evidence_refs=tuple(sorted(item.reference for item in evidence)),
    )
    return GoalRecord(
        goal_id=goal_id,
        target=target,
        description=text,
        deadline=actual_deadline,
        origin_refs=(origin,),
        evidence_refs=evidence,
        revision_history=(genesis,),
    )


def _make_goal_system(
    lifecycles: tuple[GoalLifecycle, ...],
    *,
    description: str | None = None,
    deadline: Deadline | None = None,
) -> GoalSystem:
    system = GoalSystem()
    proposals = tuple(
        _goal_proposal(index, description=description, deadline=deadline)
        for index in range(len(lifecycles))
    )
    for index, proposal in enumerate(proposals):
        system.ingest_proposal(
            proposal,
            _goal_event(
                len(proposals) + index + 1,
                proposal.evidence_refs,
                event_id=f"event:goal:ingest:{index:03}",
            ),
        )

    for index, (proposal, lifecycle) in enumerate(zip(proposals, lifecycles, strict=True)):
        if lifecycle is GoalLifecycle.PROPOSED:
            continue
        sequence = 2 * len(proposals) + index + 1
        admission_event = _goal_event(
            sequence,
            proposal.evidence_refs,
            event_id=f"event:goal:admit:{index:03}",
        )
        admission = GoalSubjectAdmission(
            goal_id=proposal.goal_id,
            proposal_digest=proposal.proposal_digest,
            evidence_refs=tuple(sorted(item.reference for item in proposal.evidence_refs)),
            event_id=admission_event.event_id,
            event_sequence=admission_event.event_sequence,
            reason=GoalAdmissionReason.SUBJECT_ENDORSEMENT,
        )
        system.adopt(proposal.goal_id, admission, admission_event)

    for index, (proposal, lifecycle) in enumerate(zip(proposals, lifecycles, strict=True)):
        if lifecycle not in {GoalLifecycle.DEFERRED, GoalLifecycle.ABANDONED}:
            continue
        sequence = 3 * len(proposals) + index + 1
        event_refs = (R13Reference(R13ReferenceKind.EVENT, f"event:goal:decision:{index:03}"),)
        event = _goal_event(
            sequence,
            event_refs,
            event_id=f"event:goal:decision:{index:03}",
        )
        operation, reason = (
            (GoalRevisionOperation.DEFER, GoalRevisionReason.SUBJECT_DEFERRED)
            if lifecycle is GoalLifecycle.DEFERRED
            else (GoalRevisionOperation.ABANDON, GoalRevisionReason.SUBJECT_ABANDONED)
        )
        proof = GoalSubjectTransitionProof(
            goal_id=proposal.goal_id,
            proposal_digest=proposal.proposal_digest,
            operation=operation,
            reason=reason,
            previous_lifecycle_state=GoalLifecycle.ADOPTED,
            evidence_refs=tuple(sorted(item.reference for item in event_refs)),
            event_id=event.event_id,
            event_sequence=event.event_sequence,
        )
        if lifecycle is GoalLifecycle.DEFERRED:
            system.defer(proposal.goal_id, proof, event)
        else:
            system.abandon(proposal.goal_id, proof, event)
    return system


def _commitment_proposal(
    index: int,
    *,
    scope: tuple[str, ...] | None = None,
    deadline: Deadline | None = None,
) -> CommitmentRecord:
    subject = f"private subject {index:03}"
    beneficiary = R13Reference(R13ReferenceKind.EXTERNAL_PARTY, f"party:beneficiary:{index:03}")
    actual_scope = scope or (f"scope:item:{index:03}",)
    deadline = deadline if deadline is not None else Deadline()
    evidence = (R13Reference(R13ReferenceKind.EVENT, f"event:commitment:evidence:{index:03}"),)
    origin = (R13Reference(R13ReferenceKind.USER_REQUEST, f"request:commitment:{index:03}"),)
    proposal_digest = commitment_proposal_digest(
        subject=subject,
        beneficiary=beneficiary,
        scope=actual_scope,
        deadline=deadline,
        evidence_refs=evidence,
        origin_refs=origin,
    )
    commitment_id = commitment_id_for_proposal(proposal_digest)
    sequence = index + 1
    genesis = CommitmentRevisionRecord(
        commitment_id=commitment_id,
        revision=0,
        operation=CommitmentRevisionOperation.CREATE,
        reason=CommitmentRevisionReason.CREATION,
        created_at=_at(sequence),
        previous_lifecycle_state=None,
        event_id=f"event:commitment:create:{index:03}",
        event_sequence=sequence,
        evidence_refs=tuple(sorted(item.reference for item in evidence)),
    )
    return CommitmentRecord(
        subject=subject,
        beneficiary=beneficiary,
        scope=actual_scope,
        deadline=deadline,
        evidence_refs=evidence,
        origin_refs=origin,
        revision_history=(genesis,),
    )


def _make_commitment_system(
    lifecycles: tuple[CommitmentLifecycle, ...],
    *,
    scope: tuple[str, ...] | None = None,
    deadline: Deadline | None = None,
) -> CommitmentSystem:
    system = CommitmentSystem()
    proposals = tuple(
        _commitment_proposal(index, scope=scope, deadline=deadline)
        for index in range(len(lifecycles))
    )
    for index, proposal in enumerate(proposals):
        system.ingest_proposal(
            proposal,
            _commitment_event(
                len(proposals) + index + 1,
                proposal.evidence_refs,
                event_id=f"event:commitment:ingest:{index:03}",
            ),
        )

    for index, (proposal, lifecycle) in enumerate(zip(proposals, lifecycles, strict=True)):
        if lifecycle is CommitmentLifecycle.PROPOSED:
            continue
        sequence = 2 * len(proposals) + index + 1
        event = _commitment_event(
            sequence,
            proposal.evidence_refs,
            event_id=f"event:commitment:accept:{index:03}",
        )
        admission = CommitmentSubjectAdmission(
            commitment_id=proposal.commitment_id,
            proposal_digest=proposal.proposal_digest,
            beneficiary=proposal.beneficiary,
            scope=proposal.scope,
            deadline=proposal.deadline,
            evidence_refs=tuple(sorted(item.reference for item in proposal.evidence_refs)),
            event_id=event.event_id,
            event_sequence=event.event_sequence,
            reason=CommitmentAdmissionReason.SUBJECT_ENDORSEMENT,
        )
        system.accept(proposal.commitment_id, admission, event)

    for index, (proposal, lifecycle) in enumerate(zip(proposals, lifecycles, strict=True)):
        if lifecycle not in {CommitmentLifecycle.RELEASED, CommitmentLifecycle.RENEGOTIATED}:
            continue
        sequence = 3 * len(proposals) + index + 1
        event_refs = (
            R13Reference(R13ReferenceKind.EVENT, f"event:commitment:decision:{index:03}"),
        )
        event = _commitment_event(
            sequence,
            event_refs,
            event_id=f"event:commitment:decision:{index:03}",
        )
        operation, reason = (
            (CommitmentRevisionOperation.RELEASE, CommitmentRevisionReason.SUBJECT_RELEASE)
            if lifecycle is CommitmentLifecycle.RELEASED
            else (
                CommitmentRevisionOperation.RENEGOTIATE,
                CommitmentRevisionReason.SUBJECT_RENEGOTIATION,
            )
        )
        proof = CommitmentSubjectTransitionProof(
            commitment_id=proposal.commitment_id,
            proposal_digest=proposal.proposal_digest,
            beneficiary=proposal.beneficiary,
            scope=proposal.scope,
            deadline=proposal.deadline,
            operation=operation,
            reason=reason,
            previous_lifecycle_state=CommitmentLifecycle.ACTIVE,
            evidence_refs=tuple(sorted(item.reference for item in event_refs)),
            event_id=event.event_id,
            event_sequence=event.event_sequence,
        )
        if lifecycle is CommitmentLifecycle.RELEASED:
            system.release(proposal.commitment_id, proof, event)
        else:
            system.renegotiate(proposal.commitment_id, proof, event)
    return system


def _terminal_goal(
    record: GoalRecord,
    lifecycle: GoalLifecycle,
) -> GoalRecord:
    if lifecycle not in {GoalLifecycle.COMPLETED, GoalLifecycle.FAILED}:
        raise AssertionError("terminal Goal helper requires a terminal lifecycle")
    sequence = record.revision_history[-1].event_sequence
    assert sequence is not None
    sequence += 1
    operation = (
        GoalRevisionOperation.COMPLETE
        if lifecycle is GoalLifecycle.COMPLETED
        else GoalRevisionOperation.FAIL
    )
    outcome = R13Reference(R13ReferenceKind.EXPERIENCE, f"experience:goal:outcome:{lifecycle.value}")
    revision = GoalRevisionRecord(
        goal_id=record.goal_id,
        revision=record.revision + 1,
        operation=operation,
        reason=GoalRevisionReason.VERIFIED_OUTCOME,
        created_at=_at(sequence),
        previous_lifecycle_state=GoalLifecycle.ADOPTED,
        proposal_digest=record.proposal_digest,
        event_id=f"event:goal:outcome:{lifecycle.value}",
        event_sequence=sequence,
        evidence_refs=(outcome.reference,),
        previous_revision_digest=record.revision_history[-1].record_digest,
    )
    return replace(
        record,
        lifecycle=lifecycle,
        outcome_evidence_refs=(outcome,),
        revision=record.revision + 1,
        revision_history=(*record.revision_history, revision),
    )


def _terminal_commitment(
    record: CommitmentRecord,
    lifecycle: CommitmentLifecycle,
) -> CommitmentRecord:
    if lifecycle not in {CommitmentLifecycle.FULFILLED, CommitmentLifecycle.BREACHED}:
        raise AssertionError("terminal Commitment helper requires a terminal lifecycle")
    sequence = record.revision_history[-1].event_sequence
    assert sequence is not None
    sequence += 1
    operation = (
        CommitmentRevisionOperation.FULFILL
        if lifecycle is CommitmentLifecycle.FULFILLED
        else CommitmentRevisionOperation.BREACH
    )
    outcome = R13Reference(
        R13ReferenceKind.EXPERIENCE,
        f"experience:commitment:outcome:{lifecycle.value}",
    )
    revision = CommitmentRevisionRecord(
        commitment_id=record.commitment_id,
        revision=record.revision + 1,
        operation=operation,
        reason=CommitmentRevisionReason.VERIFIED_OUTCOME,
        created_at=_at(sequence),
        previous_lifecycle_state=CommitmentLifecycle.ACTIVE,
        event_id=f"event:commitment:outcome:{lifecycle.value}",
        event_sequence=sequence,
        evidence_refs=(outcome.reference,),
        previous_revision_digest=record.revision_history[-1].record_digest,
    )
    return replace(
        record,
        lifecycle=lifecycle,
        outcome_evidence_refs=(outcome,),
        revision=record.revision + 1,
        revision_history=(*record.revision_history, revision),
    )


def test_projection_includes_all_active_records_and_derives_exact_size() -> None:
    assert R13_PROMPT_MAX_RECORDS_PER_DOMAIN == 32
    count = R13_PROMPT_MAX_RECORDS_PER_DOMAIN
    motivation = _make_motivation_system((MotivationLifecycle.ACTIVE,) * count)
    goal = _make_goal_system((GoalLifecycle.ADOPTED,) * count)
    commitment = _make_commitment_system((CommitmentLifecycle.ACTIVE,) * count)
    view = R13PromptView.from_snapshots(
        motivation.snapshot(), goal.snapshot(), commitment.snapshot()
    )

    assert len(view.motivations) == count
    assert len(view.goals) == count
    assert len(view.commitments) == count
    assert tuple(entry.motivation_id for entry in view.motivations) == tuple(
        sorted(entry.motivation_id for entry in view.motivations)
    )
    assert tuple(entry.goal_id for entry in view.goals) == tuple(
        sorted(entry.goal_id for entry in view.goals)
    )
    assert tuple(entry.commitment_id for entry in view.commitments) == tuple(
        sorted(entry.commitment_id for entry in view.commitments)
    )
    assert view.serialized_bytes == len(canonical_json(view.canonical_value()))
    assert view.serialized_bytes <= R13_PROMPT_MAX_SERIALIZED_BYTES
    assert len(view.render().encode("utf-8")) <= R13_PROMPT_MAX_RENDERED_BYTES

    with pytest.raises(ValueError, match="motivations.*bound"):
        R13PromptView(motivations=(view.motivations[0],) * (count + 1))
    with pytest.raises(ValueError, match="goals.*bound"):
        R13PromptView(goals=(view.goals[0],) * (count + 1))
    with pytest.raises(ValueError, match="commitments.*bound"):
        R13PromptView(commitments=(view.commitments[0],) * (count + 1))


def test_projection_excludes_each_noncurrent_lifecycle_and_all_proofs() -> None:
    motivation_statuses = (
        MotivationLifecycle.ACTIVE,
        MotivationLifecycle.DORMANT,
        MotivationLifecycle.SATIATED,
        MotivationLifecycle.RETIRED,
    )
    goal_statuses = (
        GoalLifecycle.PROPOSED,
        GoalLifecycle.ADOPTED,
        GoalLifecycle.DEFERRED,
        GoalLifecycle.ABANDONED,
    )
    commitment_statuses = (
        CommitmentLifecycle.PROPOSED,
        CommitmentLifecycle.ACTIVE,
        CommitmentLifecycle.RELEASED,
        CommitmentLifecycle.RENEGOTIATED,
    )
    motivation = _make_motivation_system(motivation_statuses)
    goal = _make_goal_system(goal_statuses)
    commitment = _make_commitment_system(commitment_statuses)
    before = (
        motivation.snapshot().authority_digest,
        goal.snapshot().authority_digest,
        commitment.snapshot().authority_digest,
    )

    view = R13PromptView.from_snapshots(
        motivation.snapshot(), goal.snapshot(), commitment.snapshot()
    )

    assert len(view.motivations) == 1
    assert view.motivations[0].motivation_id == next(
        record.motivation_id
        for record in motivation.records
        if record.lifecycle is MotivationLifecycle.ACTIVE
    )
    assert len(view.goals) == 1
    assert view.goals[0].goal_id == next(
        record.goal_id for record in goal.records if record.lifecycle is GoalLifecycle.ADOPTED
    )
    assert len(view.commitments) == 1
    assert view.commitments[0].commitment_id == next(
        record.commitment_id
        for record in commitment.records
        if record.lifecycle is CommitmentLifecycle.ACTIVE
    )
    assert before == (
        motivation.snapshot().authority_digest,
        goal.snapshot().authority_digest,
        commitment.snapshot().authority_digest,
    )

    for record in motivation.records:
        if record.lifecycle is not MotivationLifecycle.ACTIVE:
            with pytest.raises(ValueError, match="active Motivations"):
                MotivationPromptEntry._from_record(record)
    for record in goal.records:
        if record.lifecycle is not GoalLifecycle.ADOPTED:
            with pytest.raises(ValueError, match="adopted Goals"):
                GoalPromptEntry._from_record(record)
    for record in commitment.records:
        if record.lifecycle is not CommitmentLifecycle.ACTIVE:
            with pytest.raises(ValueError, match="active Commitments"):
                CommitmentPromptEntry._from_record(record)

    goal_adopted = next(
        record for record in goal.records if record.lifecycle is GoalLifecycle.ADOPTED
    )
    commitment_active = next(
        record
        for record in commitment.records
        if record.lifecycle is CommitmentLifecycle.ACTIVE
    )
    for lifecycle in (GoalLifecycle.COMPLETED, GoalLifecycle.FAILED):
        with pytest.raises(ValueError, match="adopted Goals"):
            GoalPromptEntry._from_record(_terminal_goal(goal_adopted, lifecycle))
    for lifecycle in (CommitmentLifecycle.FULFILLED, CommitmentLifecycle.BREACHED):
        with pytest.raises(ValueError, match="active Commitments"):
            CommitmentPromptEntry._from_record(
                _terminal_commitment(commitment_active, lifecycle)
            )

    canonical = view.canonical_value()
    assert canonical["schema_version"] == R13_PROMPT_SCHEMA_VERSION
    assert set(canonical) == {"schema_version", "motivations", "goals", "commitments"}
    assert set(canonical["motivations"][0]) == {
        "motivation_id",
        "kind",
        "target",
        "strength",
        "persistence",
        "satiation",
        "uncertainty",
    }
    assert set(canonical["goals"][0]) == {
        "goal_id",
        "target",
        "description",
        "deadline",
    }
    assert set(canonical["commitments"][0]) == {
        "commitment_id",
        "beneficiary",
        "scope",
        "deadline",
    }
    assert "subject" not in canonical["commitments"][0]
    assert not any(
        "evidence" in key or "proof" in key or "history" in key or "outcome" in key
        for entries in (canonical["motivations"], canonical["goals"], canonical["commitments"])
        for entry in entries
        for key in entry
    )


def test_view_rejects_noncanonical_ordering_wrong_types_and_manual_entries() -> None:
    motivation = _make_motivation_system((MotivationLifecycle.ACTIVE,) * 2)
    goal = _make_goal_system((GoalLifecycle.ADOPTED,) * 2)
    commitment = _make_commitment_system((CommitmentLifecycle.ACTIVE,) * 2)
    view = R13PromptView.from_snapshots(
        motivation.snapshot(), goal.snapshot(), commitment.snapshot()
    )

    with pytest.raises(ValueError, match="sorted by unique"):
        R13PromptView(motivations=tuple(reversed(view.motivations)))
    with pytest.raises(ValueError, match="sorted by unique"):
        R13PromptView(goals=tuple(reversed(view.goals)))
    with pytest.raises(ValueError, match="sorted by unique"):
        R13PromptView(commitments=tuple(reversed(view.commitments)))
    with pytest.raises(TypeError, match="exact tuple"):
        R13PromptView(motivations=list(view.motivations))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact MotivationPromptEntry"):
        R13PromptView(motivations=(motivation.records[0],))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="created by R13PromptView"):
        MotivationPromptEntry()
    with pytest.raises(TypeError, match="created by R13PromptView"):
        GoalPromptEntry()
    with pytest.raises(TypeError, match="created by R13PromptView"):
        CommitmentPromptEntry()

    with pytest.raises(TypeError, match="exact MotivationSystemSnapshot"):
        R13PromptView.from_snapshots(
            GoalSystemSnapshot(), goal.snapshot(), commitment.snapshot()  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exact GoalSystemSnapshot"):
        R13PromptView.from_snapshots(
            motivation.snapshot(), CommitmentSystemSnapshot(), commitment.snapshot()  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exact CommitmentSystemSnapshot"):
        R13PromptView.from_snapshots(
            motivation.snapshot(), goal.snapshot(), GoalSystemSnapshot()  # type: ignore[arg-type]
        )


def test_snapshot_and_entry_fields_are_revalidated_without_retaining_authority() -> None:
    motivation = _make_motivation_system((MotivationLifecycle.ACTIVE,))
    goal = _make_goal_system((GoalLifecycle.ADOPTED,))
    commitment = _make_commitment_system((CommitmentLifecycle.ACTIVE,))
    motivation_snapshot = motivation.snapshot()
    goal_snapshot = goal.snapshot()
    commitment_snapshot = commitment.snapshot()

    for bad_snapshot in (motivation_snapshot, goal_snapshot, commitment_snapshot):
        object.__setattr__(bad_snapshot, "schema_version", 99)
        with pytest.raises(ValueError, match="snapshot version"):
            R13PromptView.from_snapshots(
                motivation_snapshot, goal_snapshot, commitment_snapshot
            )
        object.__setattr__(bad_snapshot, "schema_version", 1)

    # Use fresh snapshots after the deliberate frozen-object tamper above.
    view = R13PromptView.from_snapshots(
        motivation.snapshot(), goal.snapshot(), commitment.snapshot()
    )
    motivation_entry = view.motivations[0]
    object.__setattr__(motivation_entry, "strength", float("nan"))
    with pytest.raises(ValueError, match="strength must be finite"):
        R13PromptView(motivations=(motivation_entry,))

    expected_fields = {
        MotivationPromptEntry: {
            "motivation_id",
            "kind",
            "target",
            "strength",
            "persistence",
            "satiation",
            "uncertainty",
        },
        GoalPromptEntry: {"goal_id", "target", "description", "deadline"},
        CommitmentPromptEntry: {
            "commitment_id",
            "beneficiary",
            "scope",
            "deadline",
        },
    }
    for entry_type, expected in expected_fields.items():
        assert {item.name for item in fields(entry_type)} == expected

    for value in (view, *view.motivations, *view.goals, *view.commitments):
        with pytest.raises((AttributeError, TypeError)):
            value.extra_mutable_field = []  # type: ignore[attr-defined,misc]


def test_extreme_unicode_is_escaped_and_output_never_shortens_records() -> None:
    unicode_scalar = "\U00010000"
    count = R13_PROMPT_MAX_RECORDS_PER_DOMAIN
    description = unicode_scalar * 1_024
    scope = tuple(
        chr(0x10000 + index) + unicode_scalar * 255
        for index in range(32)
    )
    deadline = Deadline(datetime.max.replace(tzinfo=UTC))
    assert deadline.canonical_value() == "9999-12-31T23:59:59.999999Z"
    motivation = _make_motivation_system(
        (MotivationLifecycle.ACTIVE,) * count,
        salience=1.0,
        confidence=1.0,
        persistence=1.0,
        uncertainty=1.0,
    )
    goal = _make_goal_system(
        (GoalLifecycle.ADOPTED,) * count, description=description, deadline=deadline
    )
    commitment = _make_commitment_system(
        (CommitmentLifecycle.ACTIVE,) * count, scope=scope, deadline=deadline
    )
    view = R13PromptView.from_snapshots(
        motivation.snapshot(), goal.snapshot(), commitment.snapshot()
    )

    rendered = view.render()
    lines = rendered.splitlines()
    assert lines[0] == (
        "A current Motivation is not a Goal; an adopted Goal is not a Commitment. "
        "An active Commitment is accepted responsibility."
    )
    assert lines.count("Current Motivations:") == 1
    assert lines.count("Adopted Goals:") == 1
    assert lines.count("Active Commitments:") == 1
    assert "\U00010000" not in rendered
    assert "\\ud800\\udc00" in rendered
    assert len(rendered.encode("utf-8")) <= R13_PROMPT_MAX_RENDERED_BYTES
    assert len(canonical_json(view.canonical_value())) <= R13_PROMPT_MAX_SERIALIZED_BYTES
    assert len(view.motivations) == len(view.goals) == len(view.commitments) == count
    assert view.goals[0].description == description
    assert view.commitments[0].scope == scope
    assert view.goals[0].deadline == deadline
    assert view.commitments[0].deadline == deadline

    goal_line = lines[lines.index("Adopted Goals:") + 1]
    commitment_line = lines[lines.index("Active Commitments:") + 1]
    assert json.loads(goal_line)["description"] == description
    assert json.loads(commitment_line)["scope"] == list(scope)
    assert json.loads(goal_line)["deadline"] == deadline.canonical_value()
    assert json.loads(commitment_line)["deadline"] == deadline.canonical_value()


def test_prompt_view_rejects_unprojected_domain_objects_and_bad_bounds() -> None:
    goal = _goal_proposal(0)
    commitment = _commitment_proposal(0)
    with pytest.raises(TypeError, match="exact GoalPromptEntry"):
        R13PromptView(goals=(goal,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact CommitmentPromptEntry"):
        R13PromptView(commitments=(commitment,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact GoalPromptEntry"):
        R13PromptView(goals=(goal.revision_history[0],))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact CommitmentPromptEntry"):
        R13PromptView(
            commitments=(commitment.revision_history[0],)  # type: ignore[arg-type]
        )

    motivation_system = _make_motivation_system((MotivationLifecycle.ACTIVE,))
    evidence = motivation_system.snapshot().evidence_ledger[0].evidence
    with pytest.raises(TypeError, match="exact MotivationPromptEntry"):
        R13PromptView(motivations=(evidence,))  # type: ignore[arg-type]

    goal_system = _make_goal_system((GoalLifecycle.ADOPTED,))
    commitment_system = _make_commitment_system((CommitmentLifecycle.ACTIVE,))
    goal_admission = goal_system.records[0].subject_admission
    commitment_admission = commitment_system.records[0].subject_admission
    assert goal_admission is not None
    assert commitment_admission is not None
    with pytest.raises(TypeError, match="exact GoalPromptEntry"):
        R13PromptView(goals=(goal_admission,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact CommitmentPromptEntry"):
        R13PromptView(commitments=(commitment_admission,))  # type: ignore[arg-type]
    goal_entry = GoalPromptEntry._from_record(goal_system.records[0])
    commitment_entry = CommitmentPromptEntry._from_record(commitment_system.records[0])
    object.__setattr__(goal_entry, "description", "line one\nline two")
    with pytest.raises(ValueError, match="canonical text"):
        R13PromptView(goals=(goal_entry,))
    object.__setattr__(commitment_entry, "scope", ("z", "a"))
    with pytest.raises(ValueError, match="canonically ordered"):
        R13PromptView(commitments=(commitment_entry,))


def test_projection_imports_without_runtime_or_model_stacks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import suzka.motivation.projection; "
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
