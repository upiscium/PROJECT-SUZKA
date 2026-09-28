"""Focused tests for the intrinsic U5 Belief authority."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from suzka.belief import (
    AdmissionReason,
    BELIEF_MAX_RECORDS,
    BeliefCapacityExceeded,
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


def full_evidence(prefix: str) -> tuple[BeliefEvidence, ...]:
    return tuple(
        BeliefEvidence(f"{prefix}-{index:02d}", BeliefEvidenceType.EXPERIENCE)
        for index in range(32)
    )


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


def test_correction_can_replace_evidence_with_a_fresh_admission() -> None:
    system = make_system()
    proposed = system.create_proposal(
        BeliefProposition("correctable claim"),
        event(1),
        evidence=(BeliefEvidence("claim:old", BeliefEvidenceType.EXTERNAL_CLAIM),),
    )
    adopted = system.adopt(
        proposed.belief_id,
        BeliefSubjectAdmission(
            proposed.proposition.proposition_digest,
            ("claim:old",),
            "event:2",
            2,
            AdmissionReason.SUBJECT_ENDORSEMENT,
        ),
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
    )
    fresh_admission = BeliefSubjectAdmission(
        adopted.proposition.proposition_digest,
        ("claim:new",),
        "event:3",
        3,
        AdmissionReason.SUBJECT_CORRECTION,
    )
    corrected = system.correct(
        adopted.belief_id,
        replace(
            adopted,
            confidence=0.9,
            evidence=(BeliefEvidence("claim:new", BeliefEvidenceType.EXPERIENCE),),
            subject_admission=fresh_admission,
        ),
        event(3),
        admission=fresh_admission,
    )

    assert tuple(item.evidence_ref for item in corrected.evidence) == ("claim:new",)
    assert corrected.subject_admission == fresh_admission
    assert corrected.revision_history[-1].operation is BeliefRevisionOperation.CORRECT
    assert fresh_admission.admission_digest in corrected.revision_history[-1].evidence_refs


@pytest.mark.parametrize("terminal_operation", ["retract", "expire"])
def test_full_domain_evidence_survives_adopt_correct_and_terminal_mutations(
    terminal_operation: str,
) -> None:
    system = make_system()
    evidence = full_evidence("full")
    proposed = system.create_proposal(
        BeliefProposition(f"full-evidence-{terminal_operation}"),
        event(1),
        evidence=evidence,
    )
    adopted = system.adopt(
        proposed.belief_id,
        BeliefSubjectAdmission(
            proposed.proposition.proposition_digest,
            tuple(item.evidence_ref for item in evidence),
            "event:2",
            2,
            AdmissionReason.SUBJECT_ENDORSEMENT,
        ),
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
    )
    assert len(adopted.revision_history[-1].evidence_refs) == 33

    correction_admission = BeliefSubjectAdmission(
        adopted.proposition.proposition_digest,
        tuple(item.evidence_ref for item in evidence),
        "event:3",
        3,
        AdmissionReason.SUBJECT_CORRECTION,
    )
    corrected = system.correct(
        adopted.belief_id,
        replace(adopted, confidence=0.9, subject_admission=correction_admission),
        event(3),
        admission=correction_admission,
    )
    assert len(corrected.revision_history[-1].evidence_refs) == 33

    terminal_admission = BeliefSubjectAdmission(
        corrected.proposition.proposition_digest,
        tuple(item.evidence_ref for item in evidence),
        "event:4",
        4,
        AdmissionReason.SUBJECT_REVIEW,
    )
    terminal = getattr(system, terminal_operation)(
        corrected.belief_id,
        event(4),
        admission=terminal_admission,
    )
    assert len(terminal.revision_history[-1].evidence_refs) == 33


def test_supersession_retains_two_admission_proofs_beyond_domain_evidence_bound() -> None:
    system = make_system()
    predecessor_evidence = full_evidence("predecessor")
    successor_evidence = full_evidence("successor")
    predecessor = system.create_proposal(
        BeliefProposition("full predecessor"), event(1), evidence=predecessor_evidence
    )
    successor_proposition = BeliefProposition("full successor")
    successor = BeliefRecord(
        belief_id=belief_id_for_proposition(successor_proposition.proposition_digest),
        proposition=successor_proposition,
        lifecycle=BeliefLifecycle.PROPOSED,
        epistemic_status=BeliefEpistemicStatus.UNKNOWN,
        confidence=0.5,
        evidence=successor_evidence,
        supersedes_id=predecessor.belief_id,
    )
    predecessor_admission = BeliefSubjectAdmission(
        predecessor.proposition.proposition_digest,
        tuple(item.evidence_ref for item in predecessor_evidence),
        "event:2",
        2,
        AdmissionReason.SUBJECT_REVIEW,
    )
    successor_admission = BeliefSubjectAdmission(
        successor.proposition.proposition_digest,
        tuple(item.evidence_ref for item in successor_evidence),
        "event:2",
        2,
        AdmissionReason.SUBJECT_REVIEW,
    )

    superseded, created = system.supersede(
        predecessor.belief_id,
        successor,
        event(2),
        predecessor_admission=predecessor_admission,
        successor_admission=successor_admission,
    )

    assert len(superseded.revision_history[-1].evidence_refs) == 34
    assert len(created.revision_history[-1].evidence_refs) == 33


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
    predecessor_admission = BeliefSubjectAdmission(
        adopted.proposition.proposition_digest,
        ("claim:1",),
        "event:6",
        6,
        AdmissionReason.SUBJECT_REVIEW,
    )
    successor_admission = BeliefSubjectAdmission(
        successor.proposition.proposition_digest,
        ("claim:replacement",),
        "event:6",
        6,
        AdmissionReason.SUBJECT_REVIEW,
    )
    with pytest.raises(BeliefDomainError, match="subject admission"):
        system.supersede(
            adopted.belief_id,
            successor,
            event(6),
            successor_admission=successor_admission,
        )
    with pytest.raises(BeliefDomainError, match="subject admission"):
        system.supersede(
            adopted.belief_id,
            successor,
            event(6),
            predecessor_admission=predecessor_admission,
        )

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
    predecessor_admission = BeliefSubjectAdmission(
        corrected.proposition.proposition_digest,
        ("event:1",),
        "event:4",
        4,
        AdmissionReason.SUBJECT_REVIEW,
    )
    successor_admission = BeliefSubjectAdmission(
        successor.proposition.proposition_digest,
        ("event:4",),
        "event:4",
        4,
        AdmissionReason.SUBJECT_REVIEW,
    )
    superseded, created = system.supersede(
        corrected.belief_id,
        successor,
        event(4),
        predecessor_admission=predecessor_admission,
        successor_admission=successor_admission,
    )

    assert superseded.lifecycle is BeliefLifecycle.SUPERSEDED
    assert superseded.superseded_by_id == created.belief_id
    assert created.supersedes_id == superseded.belief_id
    assert created.lifecycle is BeliefLifecycle.ADOPTED
    assert created.revision == 1
    assert created.revision_history[-1].operation.value == "adopt"
    supersede_revision = superseded.revision_history[-1]
    assert supersede_revision.operation is BeliefRevisionOperation.SUPERSEDE
    assert predecessor_admission.admission_digest in supersede_revision.evidence_refs
    assert successor_admission.admission_digest in supersede_revision.evidence_refs
    assert predecessor_admission.admission_digest != successor_admission.admission_digest
    assert system.get(created.belief_id) == created
    assert tuple(record.belief_id for record in system.records) == tuple(
        sorted(record.belief_id for record in system.records)
    )


def test_restore_rejects_supersession_admissions_from_different_events() -> None:
    system = make_system()
    predecessor = system.create_proposal(
        BeliefProposition("predecessor"),
        event(1),
        evidence=(BeliefEvidence("claim:predecessor", BeliefEvidenceType.EXPERIENCE),),
    )
    predecessor = system.adopt(
        predecessor.belief_id,
        BeliefSubjectAdmission(
            predecessor.proposition.proposition_digest,
            ("claim:predecessor",),
            "event:2",
            2,
            AdmissionReason.SUBJECT_REVIEW,
        ),
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
    )
    successor_proposition = BeliefProposition("successor")
    successor = BeliefRecord(
        belief_id=belief_id_for_proposition(successor_proposition.proposition_digest),
        proposition=successor_proposition,
        lifecycle=BeliefLifecycle.PROPOSED,
        epistemic_status=BeliefEpistemicStatus.UNKNOWN,
        confidence=0.5,
        evidence=(BeliefEvidence("claim:successor", BeliefEvidenceType.EXPERIENCE),),
        supersedes_id=predecessor.belief_id,
    )
    predecessor_admission = BeliefSubjectAdmission(
        predecessor.proposition.proposition_digest,
        ("claim:predecessor",),
        "event:3",
        3,
        AdmissionReason.SUBJECT_REVIEW,
    )
    successor_admission = BeliefSubjectAdmission(
        successor.proposition.proposition_digest,
        ("claim:successor",),
        "event:3",
        3,
        AdmissionReason.SUBJECT_REVIEW,
    )
    superseded, created = system.supersede(
        predecessor.belief_id,
        successor,
        event(3),
        predecessor_admission=predecessor_admission,
        successor_admission=successor_admission,
    )

    tampered_admission = BeliefSubjectAdmission(
        created.proposition.proposition_digest,
        ("claim:successor",),
        "event:4",
        4,
        AdmissionReason.SUBJECT_REVIEW,
    )
    tampered_successor_revision = replace(
        created.revision_history[-1],
        event_id=tampered_admission.event_id,
        event_sequence=tampered_admission.event_sequence,
        evidence_refs=tuple(
            sorted(
                ref
                for ref in (
                    *created.revision_history[-1].evidence_refs,
                    tampered_admission.admission_digest,
                )
                if ref != successor_admission.admission_digest
            )
        ),
    )
    tampered_successor = replace(
        created,
        subject_admission=tampered_admission,
        revision_history=(created.revision_history[0], tampered_successor_revision),
    )
    tampered_predecessor_revision = replace(
        superseded.revision_history[-1],
        evidence_refs=tuple(
            sorted(
                ref
                for ref in (
                    *superseded.revision_history[-1].evidence_refs,
                    tampered_admission.admission_digest,
                )
                if ref != successor_admission.admission_digest
            )
        ),
    )
    tampered_predecessor = replace(
        superseded,
        revision_history=(*superseded.revision_history[:-1], tampered_predecessor_revision),
    )
    snapshot = BeliefSystemSnapshot(
        tuple(sorted((tampered_predecessor, tampered_successor), key=lambda item: item.belief_id))
    )

    with pytest.raises(BeliefDomainError, match="share the SUPERSEDE event"):
        BeliefSystem.restore_snapshot(snapshot)


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
    first = system.create_proposal(
        BeliefProposition("first"),
        event(1),
        evidence=(BeliefEvidence("first:1", BeliefEvidenceType.EXPERIENCE),),
    )
    second = system.create_proposal(
        BeliefProposition("second"),
        event(2),
        evidence=(BeliefEvidence("second:1", BeliefEvidenceType.EXPERIENCE),),
    )

    first_admission = BeliefSubjectAdmission(
        first.proposition.proposition_digest,
        ("first:1",),
        "event:3",
        3,
        AdmissionReason.SUBJECT_REVIEW,
    )
    second_admission = BeliefSubjectAdmission(
        second.proposition.proposition_digest,
        ("second:1",),
        "event:3",
        3,
        AdmissionReason.SUBJECT_REVIEW,
    )

    def superseded(
        record: BeliefRecord,
        successor_id: str,
        admission: BeliefSubjectAdmission,
        successor_admission: BeliefSubjectAdmission,
    ) -> BeliefRecord:
        revision = BeliefRevisionRecord(
            belief_id=record.belief_id,
            revision=1,
            operation=BeliefRevisionOperation.SUPERSEDE,
            reason=BeliefRevisionReason.SUPERSESSION,
            created_at=NOW,
            previous_revision_digest=record.revision_history[0].record_digest,
            event_id=admission.event_id,
            event_sequence=admission.event_sequence,
            evidence_refs=tuple(
                sorted((admission.admission_digest, successor_admission.admission_digest))
            ),
        )
        return replace(
            record,
            lifecycle=BeliefLifecycle.SUPERSEDED,
            subject_admission=admission,
            supersedes_id=successor_id,
            superseded_by_id=successor_id,
            revision=1,
            revision_history=(record.revision_history[0], revision),
        )

    first = superseded(first, second.belief_id, first_admission, second_admission)
    second = superseded(second, first.belief_id, second_admission, first_admission)
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


def test_total_record_capacity_fails_closed_without_eviction() -> None:
    system = make_system()
    for sequence in range(1, BELIEF_MAX_RECORDS + 1):
        system.create_proposal(BeliefProposition(f"claim-{sequence}"), event(sequence))

    before = system.snapshot()
    with pytest.raises(BeliefCapacityExceeded, match="record bound"):
        system.create_proposal(
            BeliefProposition(f"claim-{BELIEF_MAX_RECORDS + 1}"),
            event(BELIEF_MAX_RECORDS + 1),
        )

    assert system.snapshot() == before


@pytest.mark.parametrize(
    ("operation", "lifecycle"),
    [
        ("retract", BeliefLifecycle.RETRACTED),
        ("expire", BeliefLifecycle.EXPIRED),
    ],
)
def test_successful_terminal_mutations_leave_the_ordinary_active_view(
    operation: str, lifecycle: BeliefLifecycle
) -> None:
    system = make_system()
    proposed = system.create_proposal(
        BeliefProposition(f"terminal-{operation}"),
        event(1),
        evidence=(BeliefEvidence("claim:terminal", BeliefEvidenceType.EXPERIENCE),),
    )
    adopted = system.adopt(
        proposed.belief_id,
        BeliefSubjectAdmission(
            proposed.proposition.proposition_digest,
            ("claim:terminal",),
            "event:2",
            2,
            AdmissionReason.SUBJECT_REVIEW,
        ),
        event(2),
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
    )
    admission = BeliefSubjectAdmission(
        adopted.proposition.proposition_digest,
        ("claim:terminal",),
        "event:3",
        3,
        AdmissionReason.SUBJECT_REVIEW,
    )

    terminal = getattr(system, operation)(adopted.belief_id, event(3), admission=admission)

    assert terminal.lifecycle is lifecycle
    assert system.ordinary_active(at=NOW) == ()
    assert system.get(adopted.belief_id) == terminal
