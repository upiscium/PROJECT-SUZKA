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
    BeliefRevisionOperation,
    BeliefRevisionReason,
    BeliefRevisionRecord,
    BeliefSubjectAdmission,
    BeliefSystem,
    BeliefSystemSnapshot,
    belief_id_for_proposition,
)
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType


NOW = datetime(2026, 1, 1, tzinfo=UTC)
_ACTIVE_EVENT: AgentEvent | None = None


def event(sequence: int) -> BeliefMutationEvidence:
    global _ACTIVE_EVENT
    _ACTIVE_EVENT = AgentEvent(
        event_id=f"event:{sequence}",
        event_type=AgentEventType.CHAT,
        source=AgentEventSource.API_CHAT,
        requested_at=NOW,
        processing_sequence=sequence,
    )
    return BeliefMutationEvidence(f"event:{sequence}", sequence, NOW)


def make_system() -> BeliefSystem:
    return BeliefSystem(event_provider=lambda: _ACTIVE_EVENT)


def test_proposal_adoption_and_active_projection_require_subject_admission() -> None:
    system = make_system()
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
    system = make_system()
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
    system = make_system()
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


def test_duck_typed_event_cannot_be_used_as_runtime_authority() -> None:
    class FakeRuntimeEvent:
        event_id = "event:1"
        event_type = AgentEventType.CHAT
        source = AgentEventSource.API_CHAT
        requested_at = NOW
        processing_sequence = 1

    system = BeliefSystem(event_provider=lambda: FakeRuntimeEvent())

    with pytest.raises(BeliefDomainError, match="invalid"):
        system.create_proposal(BeliefProposition("forged runtime event"))

    assert system.records == ()


def test_every_adopted_belief_mutation_requires_fresh_subject_admission() -> None:
    system = make_system()
    proposed = system.create_proposal(
        BeliefProposition("claim"),
        event(1),
        evidence=(BeliefEvidence("claim:1", BeliefEvidenceType.EXTERNAL_CLAIM),),
    )
    adopted = system.adopt(
        proposed.belief_id,
        BeliefSubjectAdmission(
            proposed.proposition.proposition_digest,
            ("claim:1",),
            "event:2",
            2,
            AdmissionReason.SUBJECT_REVIEW,
        ),
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
    )
    before = system.snapshot()

    with pytest.raises(BeliefDomainError, match="subject admission"):
        system.correct(adopted.belief_id, replace(adopted, confidence=0.2), event(3))
    with pytest.raises(BeliefDomainError, match="subject admission"):
        system.retract(adopted.belief_id, event(4))
    with pytest.raises(BeliefDomainError, match="subject admission"):
        system.expire(adopted.belief_id, event(5))

    successor_proposition = BeliefProposition("replacement")
    successor = BeliefRecord(
        belief_id=belief_id_for_proposition(successor_proposition.proposition_digest),
        proposition=successor_proposition,
        lifecycle=BeliefLifecycle.PROPOSED,
        epistemic_status=BeliefEpistemicStatus.UNKNOWN,
        confidence=0.2,
        evidence=(BeliefEvidence("claim:replacement", BeliefEvidenceType.EXPERIENCE),),
        supersedes_id=adopted.belief_id,
    )
    with pytest.raises(BeliefDomainError, match="subject admission"):
        system.supersede(adopted.belief_id, successor, event(6))

    assert system.snapshot() == before


def test_correction_and_supersession_preserve_immutable_history() -> None:
    system = make_system()
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
        admission=BeliefSubjectAdmission(
            adopted.proposition.proposition_digest,
            ("event:1",),
            "event:3",
            3,
            AdmissionReason.SUBJECT_CORRECTION,
        ),
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
        evidence=(BeliefEvidence("event:4", BeliefEvidenceType.EXPERIENCE),),
        supersedes_id=corrected.belief_id,
    )
    admission = BeliefSubjectAdmission(
        successor.proposition.proposition_digest,
        ("event:4",),
        "event:4",
        4,
        AdmissionReason.SUBJECT_REVIEW,
    )
    superseded, created = system.supersede(
        corrected.belief_id, successor, event(4), admission=admission
    )

    assert superseded.lifecycle is BeliefLifecycle.SUPERSEDED
    assert superseded.superseded_by_id == created.belief_id
    assert created.supersedes_id == superseded.belief_id
    assert created.lifecycle is BeliefLifecycle.ADOPTED
    assert created.revision == 1
    assert created.revision_history[-1].operation.value == "adopt"
    assert system.get(created.belief_id) == created
    assert tuple(record.belief_id for record in system.records) == tuple(
        sorted(record.belief_id for record in system.records)
    )


def test_revision_history_compacts_with_an_anchor_and_keeps_current_revision() -> None:
    # A subject admission requires matching evidence; use one reference so the
    # long correction sequence remains representative.
    system = make_system()
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
            admission=BeliefSubjectAdmission(
                current.proposition.proposition_digest,
                ("claim:1",),
                f"event:{sequence}",
                sequence,
                AdmissionReason.SUBJECT_CORRECTION,
            ),
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


def test_restore_rejects_two_node_supersession_cycles() -> None:
    system = make_system()
    first = system.create_proposal(BeliefProposition("first"), event(1))
    second = system.create_proposal(BeliefProposition("second"), event(2))

    def superseded(record: BeliefRecord, successor_id: str, sequence: int) -> BeliefRecord:
        revision = BeliefRevisionRecord(
            belief_id=record.belief_id,
            revision=1,
            operation=BeliefRevisionOperation.SUPERSEDE,
            reason=BeliefRevisionReason.SUPERSESSION,
            created_at=NOW,
            previous_revision_digest=record.revision_history[0].record_digest,
            event_id=f"event:{sequence}",
            event_sequence=sequence,
        )
        return replace(
            record,
            lifecycle=BeliefLifecycle.SUPERSEDED,
            supersedes_id=successor_id,
            superseded_by_id=successor_id,
            revision=1,
            revision_history=(record.revision_history[0], revision),
        )

    first = superseded(first, second.belief_id, 3)
    second = superseded(second, first.belief_id, 4)
    snapshot = BeliefSystemSnapshot(tuple(sorted((first, second), key=lambda item: item.belief_id)))

    with pytest.raises(BeliefDomainError, match="cycle"):
        BeliefSystem.restore_snapshot(snapshot)


def test_restore_rejects_unreachable_adopted_revision_history() -> None:
    proposition = BeliefProposition("unreachable adoption")
    evidence = BeliefEvidence("evidence:unreachable", BeliefEvidenceType.EXPERIENCE)
    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        (evidence.evidence_ref,),
        "event:2",
        2,
        AdmissionReason.SUBJECT_REVIEW,
    )
    malformed = BeliefRecord(
        belief_id=belief_id_for_proposition(proposition.proposition_digest),
        proposition=proposition,
        lifecycle=BeliefLifecycle.ADOPTED,
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
        confidence=0.5,
        evidence=(evidence,),
        subject_admission=admission,
        revision_history=(
            BeliefRevisionRecord(
                belief_id=belief_id_for_proposition(proposition.proposition_digest),
                revision=0,
                operation=BeliefRevisionOperation.CORRECT,
                reason=BeliefRevisionReason.CORRECTION,
                created_at=NOW,
                event_id="event:2",
                event_sequence=2,
            ),
        ),
    )

    with pytest.raises(BeliefDomainError, match="begin with creation"):
        BeliefSystem.restore_snapshot(BeliefSystemSnapshot((malformed,)))


def test_authoritative_records_are_bounded_and_snapshot_is_detached() -> None:
    system = make_system()
    proposal = system.create_proposal(BeliefProposition("claim"), event(1))
    snapshot = system.snapshot()
    detached = BeliefSystem.restore_snapshot(snapshot)

    assert detached.snapshot() == snapshot
    assert detached.records == (proposal,)
    with pytest.raises(AttributeError):
        detached.records.append(proposal)  # type: ignore[attr-defined]
