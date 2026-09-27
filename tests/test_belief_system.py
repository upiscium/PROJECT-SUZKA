"""Focused tests for the intrinsic U5 Belief authority."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from suzka.belief import (
    AdmissionReason,
    BeliefDomainError,
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefEpistemicStatus,
    BeliefLifecycle,
    BeliefMutationEvidence,
    BeliefProposition,
    BeliefRecord,
    BeliefSubjectAdmission,
    BeliefSystem,
    belief_id_for_proposition,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def event(sequence: int) -> BeliefMutationEvidence:
    return BeliefMutationEvidence(f"event:{sequence}", sequence, NOW)


def test_proposal_adoption_and_active_projection_require_subject_admission() -> None:
    system = BeliefSystem()
    proposition = BeliefProposition("Alice likes tea", "Alice", "likes", "tea")
    proposed = system.create_proposal(
        proposition,
        event(1),
        confidence=0.6,
        evidence=(BeliefEvidence("claim:1", BeliefEvidenceType.EXTERNAL_CLAIM),),
    )

    assert proposed.belief_id == belief_id_for_proposition(proposition.proposition_digest)
    assert proposed.lifecycle is BeliefLifecycle.PROPOSED
    assert system.ordinary_active(at=NOW) == ()
    assert proposed.revision_history[-1].event_id == "event:1"

    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        ("claim:1",),
        "event:2",
        2,
        AdmissionReason.SUBJECT_ENDORSEMENT,
    )
    adopted = system.adopt(
        proposed.belief_id,
        admission,
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
        confidence=0.8,
    )

    assert adopted.lifecycle is BeliefLifecycle.ADOPTED
    assert adopted.revision == 1
    assert adopted.revision_history[-1].revision == adopted.revision
    assert system.ordinary_active(at=NOW) == (adopted,)


def test_forged_admission_and_conflict_hint_do_not_mutate_authority() -> None:
    system = BeliefSystem()
    proposed = system.create_proposal(
        BeliefProposition("claim"),
        event(1),
        evidence=(BeliefEvidence("claim:1", BeliefEvidenceType.EXTERNAL_CLAIM),),
    )
    before = system.snapshot()
    forged = BeliefSubjectAdmission(
        proposed.proposition.proposition_digest,
        ("claim:1",),
        "event:forged",
        99,
        AdmissionReason.SUBJECT_REVIEW,
    )

    with pytest.raises(BeliefDomainError):
        system.adopt(proposed.belief_id, forged, event(2))

    assert system.snapshot() == before
    assert not hasattr(system, "upsert")


def test_correction_cannot_bypass_subject_admission() -> None:
    system = BeliefSystem()
    proposed = system.create_proposal(
        BeliefProposition("claim"),
        event(1),
        evidence=(BeliefEvidence("claim:1", BeliefEvidenceType.EXTERNAL_CLAIM),),
    )
    admission = BeliefSubjectAdmission(
        proposed.proposition.proposition_digest,
        ("claim:1",),
        "event:2",
        2,
        AdmissionReason.SUBJECT_ENDORSEMENT,
    )

    with pytest.raises(BeliefDomainError):
        system.correct(
            proposed.belief_id,
            replace(
                proposed,
                lifecycle=BeliefLifecycle.ADOPTED,
                subject_admission=admission,
            ),
            event(3),
        )

    assert system.get(proposed.belief_id) == proposed


def test_correction_and_supersession_preserve_immutable_history() -> None:
    system = BeliefSystem()
    original = system.create_proposal(
        BeliefProposition("Alice likes tea", "Alice", "likes", "tea"), event(1),
        evidence=(BeliefEvidence("event:1", BeliefEvidenceType.EXPERIENCE),),
    )
    admission = BeliefSubjectAdmission(
        original.proposition.proposition_digest,
        ("event:1",),
        "event:2",
        2,
        AdmissionReason.SUBJECT_REVIEW,
    )
    adopted = system.adopt(
        original.belief_id,
        admission,
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
        confidence=0.7,
    )
    corrected = system.correct(
        adopted.belief_id,
        replace(adopted, confidence=0.9),
        event(3),
    )
    assert corrected.revision == 2
    assert len(corrected.revision_history) == 3
    assert corrected.revision_history[1].record_digest == adopted.revision_history[1].record_digest

    successor_proposition = BeliefProposition("Alice likes coffee", "Alice", "likes", "coffee")
    successor = BeliefRecord(
        belief_id=belief_id_for_proposition(successor_proposition.proposition_digest),
        proposition=successor_proposition,
        lifecycle=BeliefLifecycle.PROPOSED,
        epistemic_status=BeliefEpistemicStatus.UNKNOWN,
        confidence=0.4,
        supersedes_id=corrected.belief_id,
    )
    superseded, created = system.supersede(corrected.belief_id, successor, event(4))

    assert superseded.lifecycle is BeliefLifecycle.SUPERSEDED
    assert superseded.superseded_by_id == created.belief_id
    assert created.supersedes_id == superseded.belief_id
    assert system.get(created.belief_id) == created
    assert tuple(record.belief_id for record in system.records) == tuple(
        sorted(record.belief_id for record in system.records)
    )


def test_revision_history_compacts_with_an_anchor_and_keeps_current_revision() -> None:
    # A subject admission requires matching evidence; use one reference so the
    # long correction sequence remains representative.
    system = BeliefSystem()
    proposed = system.create_proposal(
        BeliefProposition("claim"),
        event(1),
        evidence=(BeliefEvidence("claim:1", BeliefEvidenceType.EXTERNAL_CLAIM),),
    )
    admission = BeliefSubjectAdmission(
        proposed.proposition.proposition_digest,
        ("claim:1",),
        "event:2",
        2,
        AdmissionReason.SUBJECT_REVIEW,
    )
    current = system.adopt(
        proposed.belief_id,
        admission,
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
    )

    for sequence in range(3, 42):
        current = system.correct(
            current.belief_id,
            replace(current, confidence=(sequence % 10) / 10),
            event(sequence),
        )

    assert len(current.revision_history) == 32
    assert current.revision_history[-1].revision == current.revision
    assert current.history_anchor_digest is not None
    assert (
        current.revision_history[0].previous_revision_digest
        == current.history_anchor_digest
    )
    system.validate()
    with pytest.raises(ValueError, match="full bounded suffix"):
        replace(current, revision_history=current.revision_history[1:])


def test_authoritative_records_are_bounded_and_snapshot_is_detached() -> None:
    system = BeliefSystem()
    proposal = system.create_proposal(BeliefProposition("claim"), event(1))
    snapshot = system.snapshot()
    detached = BeliefSystem.restore_snapshot(snapshot)

    assert detached.snapshot() == snapshot
    assert detached.records == (proposal,)
    with pytest.raises(AttributeError):
        detached.records.append(proposal)  # type: ignore[attr-defined]
