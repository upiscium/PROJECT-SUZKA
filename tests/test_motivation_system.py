"""Focused U2 tests for bounded Motivation authority and its evidence boundary."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import math
import subprocess
import sys

import pytest

from suzka.belief.records import (
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefEpistemicStatus,
    BeliefLifecycle,
    BeliefProposition,
    BeliefRecord,
    BeliefSubjectAdmission,
    BeliefSubjectAdmissionReason,
)
from suzka.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
)
from suzka.identity.value_system import ValueScope, ValueState
from suzka.motivation.common import (
    MotivationModelIdentity,
    R13Reference,
    R13ReferenceKind,
)
from suzka.motivation.motivation import (
    MotivationInterpretationCandidate,
    MotivationKind,
    MotivationLifecycle,
    motivation_id_for_target,
)
from suzka.motivation.system import (
    MOTIVATION_MAX_ELAPSED_SECONDS,
    MOTIVATION_SYSTEM_MAX_SERIALIZED_BYTES,
    MotivationCapacityExceeded,
    MotivationDomainError,
    MotivationEvidence,
    MotivationEvidenceConflict,
    MotivationExperienceEvidence,
    MotivationMutationEvidence,
    MotivationSystem,
)
import suzka.motivation.system as motivation_system_module


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def ref(kind: R13ReferenceKind, value: str) -> R13Reference:
    return R13Reference(kind, value)


def make_evidence(
    source: R13Reference | None = None,
    *,
    target: R13Reference | None = None,
    kind: MotivationKind = MotivationKind.DESIRE,
    salience: float = 0.8,
    confidence: float = 0.9,
    persistence: float = 0.6,
    uncertainty: float = 0.2,
    origin_refs: tuple[R13Reference, ...] | None = None,
    source_event_id: str | None = None,
    source_event_sequence: int | None = None,
    observed_at: datetime | None = None,
) -> MotivationEvidence:
    source = source or ref(R13ReferenceKind.EXPERIENCE, "experience:one")
    target = target or ref(R13ReferenceKind.VALUE, "value:health")
    origin_refs = origin_refs or (
        ref(R13ReferenceKind.USER_REQUEST, "request:health"),
    )
    return MotivationEvidence(
        source_ref=source,
        target=target,
        kind=kind,
        salience=salience,
        confidence=confidence,
        persistence=persistence,
        uncertainty=uncertainty,
        origin_refs=origin_refs,
        source_event_id=source_event_id,
        source_event_sequence=source_event_sequence,
        observed_at=observed_at,
    )


def make_event(
    evidence_refs: tuple[R13Reference, ...],
    *,
    sequence: int,
    event_id: str | None = None,
    recorded_at: datetime | None = None,
) -> MotivationMutationEvidence:
    return MotivationMutationEvidence(
        event_id=event_id or f"event:{sequence}",
        event_sequence=sequence,
        recorded_at=recorded_at or NOW + timedelta(seconds=sequence),
        evidence_refs=evidence_refs,
    )


def source_event(
    evidence: MotivationEvidence,
    sequence: int,
    *,
    event_id: str | None = None,
    recorded_at: datetime | None = None,
):
    return make_event(
        (evidence.source_ref,),
        sequence=sequence,
        event_id=event_id,
        recorded_at=recorded_at,
    )


def make_candidate(
    evidence: MotivationEvidence,
    event: MotivationMutationEvidence,
    *,
    kind: MotivationKind = MotivationKind.AVERSION,
    target: R13Reference | None = None,
    strength: float = 1.0,
) -> MotivationInterpretationCandidate:
    return MotivationInterpretationCandidate(
        source_evidence_refs=(evidence.source_ref,),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        model_identity=MotivationModelIdentity("test-provider", "test-model"),
        suggested_kind=kind,
        suggested_target=target or ref(R13ReferenceKind.STATE, "state:model-choice"),
        suggested_strength=strength,
        suggested_persistence=1.0,
        suggested_satiation=1.0,
        suggested_uncertainty=0.0,
    )


def make_value(*, active: bool = True) -> ValueState:
    origin = IdentityOrigin(
        actor=OriginActor.SELF,
        input_kind=OriginInputKind.INTERNAL_STATE,
        admission=(
            ValueAdmissionStatus.SELF_ENDORSED
            if active
            else ValueAdmissionStatus.PENDING
        ),
        source_ref="request:exercise",
        event_id="event:value-origin",
        event_sequence=1,
    )
    return ValueState(
        value_id="value:exercise",
        revision=0,
        name="exercise",
        concept=None,
        scope=ValueScope.SUBJECT,
        context_ids=(),
        polarity=1,
        strength=0.8,
        confidence=0.9,
        stability=0.7,
        protectedness=0.4,
        negotiability=0.5,
        allowed_update_rate=0.1,
        frozen=False,
        origin=origin,
        evidence_refs=("event:value-source",),
    )


def make_adopted_belief(*, adopted: bool = True) -> BeliefRecord:
    proposition = BeliefProposition("exercise supports health")
    evidence = (BeliefEvidence("request:exercise", BeliefEvidenceType.EXTERNAL_CLAIM),)
    admission = (
        BeliefSubjectAdmission(
            proposition.proposition_digest,
            (evidence[0].evidence_ref,),
            "event:belief-admission",
            3,
            BeliefSubjectAdmissionReason.SUBJECT_ENDORSEMENT,
        )
        if adopted
        else None
    )
    return BeliefRecord(
        belief_id="belief:exercise-health",
        proposition=proposition,
        lifecycle=(BeliefLifecycle.ADOPTED if adopted else BeliefLifecycle.PROPOSED),
        epistemic_status=BeliefEpistemicStatus.ESTABLISHED,
        confidence=0.85,
        evidence=evidence,
        subject_admission=admission,
    )


def test_creation_reinforcement_identity_and_exact_evidence_replay() -> None:
    system = MotivationSystem()
    first_evidence = make_evidence()
    first_event = source_event(first_evidence, 1)

    created = system.apply_evidence(first_evidence, first_event)
    assert created.motivation_id == motivation_id_for_target(
        MotivationKind.DESIRE,
        ref(R13ReferenceKind.VALUE, "value:health"),
    )
    assert created.strength == pytest.approx(0.30 * 0.8 * 0.9)
    assert created.revision == 0
    assert created.revision_history[0].event_id == first_event.event_id
    assert system.apply_evidence(first_evidence, first_event) == created
    assert system.snapshot().authority_digest == system.snapshot().authority_digest
    assert len(system.records) == 1
    assert len(system.snapshot().event_receipts) == 1

    repeated_at_new_event = source_event(first_evidence, 2)
    assert system.apply_evidence(first_evidence, repeated_at_new_event) == created
    assert len(system.snapshot().event_receipts) == 1

    second_evidence = make_evidence(
        ref(R13ReferenceKind.EXPERIENCE, "experience:two"),
        target=created.target,
    )
    updated = system.apply_evidence(second_evidence, source_event(second_evidence, 3))
    assert updated.motivation_id == created.motivation_id
    assert updated.strength > created.strength
    assert updated.revision == 1
    assert updated.evidence_refs == tuple(
        sorted(
            (first_evidence.source_ref, second_evidence.source_ref),
            key=lambda item: (item.reference, item.kind.value),
        )
    )


def test_external_origin_is_retained_and_cannot_be_relabelled_on_replay() -> None:
    system = MotivationSystem()
    external_origin = ref(R13ReferenceKind.USER_REQUEST, "request:outside")
    evidence = make_evidence(origin_refs=(external_origin,))
    system.apply_evidence(evidence, source_event(evidence, 1))
    before = system.snapshot()

    entry = before.evidence_ledger[0]
    assert entry.evidence.origin_refs == (external_origin,)
    changed_origin = make_evidence(
        evidence.source_ref,
        origin_refs=(ref(R13ReferenceKind.SYSTEM, "system:internal"),),
    )
    with pytest.raises(MotivationEvidenceConflict, match="different source content or origin"):
        system.apply_evidence(changed_origin, source_event(changed_origin, 2))
    assert system.snapshot().authority_digest == before.authority_digest


def test_untrusted_request_and_out_of_scope_source_kinds_cannot_create_motivation() -> None:
    with pytest.raises(ValueError, match="closed R13 evidence set"):
        make_evidence(source=ref(R13ReferenceKind.USER_REQUEST, "request:direct"))
    with pytest.raises(ValueError, match="closed R13 evidence set"):
        make_evidence(source=ref(R13ReferenceKind.EXTERNAL_PARTY, "party:one"))


def test_active_value_factory_preserves_origin_and_rejects_pending_value() -> None:
    evidence = MotivationEvidence.from_value(make_value())
    assert evidence.source_ref == ref(R13ReferenceKind.VALUE, "value:exercise")
    assert evidence.target == evidence.source_ref
    assert evidence.kind is MotivationKind.DESIRE
    assert evidence.origin_refs[0].reference != evidence.source_ref.reference
    assert evidence.source_event_id == "event:value-origin"
    with pytest.raises(MotivationDomainError, match="only active Values"):
        MotivationEvidence.from_value(make_value(active=False))


def test_adopted_belief_factory_requires_subject_admission_and_preserves_claim_origin() -> None:
    belief = make_adopted_belief()
    evidence = MotivationEvidence.from_belief(belief)
    assert evidence.kind is MotivationKind.INTEREST
    assert ref(R13ReferenceKind.EXTERNAL_REQUEST, "request:exercise") in evidence.origin_refs
    assert belief.subject_admission is not None
    assert evidence.authority_witness_digest == belief.subject_admission.admission_digest
    with pytest.raises(MotivationDomainError, match="subject-adopted Beliefs"):
        MotivationEvidence.from_belief(make_adopted_belief(adopted=False))


def test_full_u1_belief_evidence_bound_is_not_narrowed_by_provenance() -> None:
    proposition = BeliefProposition("many exact evidence items support this")
    evidence = tuple(
        BeliefEvidence(f"request:{index:02d}", BeliefEvidenceType.EXTERNAL_CLAIM)
        for index in range(32)
    )
    evidence_refs = tuple(sorted(item.evidence_ref for item in evidence))
    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        evidence_refs,
        "event:belief-admission",
        33,
        BeliefSubjectAdmissionReason.SUBJECT_REVIEW,
    )
    belief = BeliefRecord(
        belief_id="belief:wide-evidence",
        proposition=proposition,
        lifecycle=BeliefLifecycle.ADOPTED,
        epistemic_status=BeliefEpistemicStatus.ESTABLISHED,
        confidence=0.8,
        evidence=evidence,
        subject_admission=admission,
    )
    projected = MotivationEvidence.from_belief(belief)
    assert len(projected.origin_refs) == 32
    assert projected.authority_witness_digest == admission.admission_digest


def test_experience_factory_uses_bounded_appraisal_and_keeps_event_origin() -> None:
    projection = MotivationExperienceEvidence(
        experience_ref=ref(R13ReferenceKind.EXPERIENCE, "experience:novel"),
        source_event_id="event:experience-source",
        source_event_sequence=2,
        created_at=NOW,
        active=True,
        novelty=0.9,
        novelty_valid=True,
        goal_progress=None,
        threat=None,
        emotion_valence=0.1,
        subjective_salience=0.8,
    )
    evidence = MotivationEvidence.from_experience(projection)
    assert evidence is not None
    assert evidence.kind is MotivationKind.INTEREST
    assert evidence.origin_refs == (ref(R13ReferenceKind.EVENT, "event:experience-source"),)
    assert evidence.salience == 0.8

    invalid_novelty = replace(projection, novelty=None, novelty_valid=False)
    assert MotivationEvidence.from_experience(invalid_novelty) is None
    inactive = replace(projection, active=False)
    with pytest.raises(MotivationDomainError, match="only active Experiences"):
        MotivationEvidence.from_experience(inactive)


def test_emotion_projection_is_typed_event_bound_and_deterministic() -> None:
    evidence = MotivationEvidence.from_emotion(
        ref(R13ReferenceKind.EMOTION, "emotion:current"),
        valence=-0.7,
        arousal=0.9,
        origin_refs=(ref(R13ReferenceKind.USER_REQUEST, "request:emotion"),),
        source_event_id="event:emotion-source",
        source_event_sequence=2,
        observed_at=NOW,
    )
    assert evidence.kind is MotivationKind.AVERSION
    assert evidence.target == ref(R13ReferenceKind.STATE, "state:emotion:negative")
    assert evidence.salience == 0.9
    with pytest.raises(ValueError, match="Emotion reference"):
        MotivationEvidence.from_emotion(
            ref(R13ReferenceKind.EXPERIENCE, "experience:wrong"),
            valence=0.0,
            arousal=0.0,
            origin_refs=(ref(R13ReferenceKind.SYSTEM, "system:emotion"),),
            source_event_id="event:emotion-source",
            source_event_sequence=2,
        )


def test_candidate_alone_has_no_motivation_authority() -> None:
    system = MotivationSystem()
    evidence = make_evidence()
    event = source_event(evidence, 1)
    candidate = make_candidate(evidence, event)

    assert system.ingest_candidate(candidate, event) == candidate
    assert system.records == ()
    snapshot = system.snapshot()
    assert snapshot.candidates == (candidate,)
    assert snapshot.evidence_ledger == ()
    assert snapshot.event_receipts == ()


def test_candidate_suggestions_cannot_override_authoritative_evidence_policy() -> None:
    system = MotivationSystem()
    evidence = make_evidence(
        kind=MotivationKind.DESIRE,
        target=ref(R13ReferenceKind.VALUE, "value:health"),
        salience=0.6,
        confidence=0.8,
    )
    event = source_event(evidence, 1)
    candidate = make_candidate(
        evidence,
        event,
        kind=MotivationKind.AVERSION,
        target=ref(R13ReferenceKind.STATE, "state:model-choice"),
        strength=1.0,
    )

    record = system.apply_evidence(evidence, event, candidate=candidate)
    assert record.kind is MotivationKind.DESIRE
    assert record.target == ref(R13ReferenceKind.VALUE, "value:health")
    assert record.strength == pytest.approx(0.30 * 0.6 * 0.8)
    assert system.snapshot().candidates == (candidate,)


def test_candidate_cannot_be_added_by_replaying_an_already_committed_event() -> None:
    system = MotivationSystem()
    evidence = make_evidence()
    event = source_event(evidence, 1)
    system.apply_evidence(evidence, event)
    before = system.snapshot()
    candidate = make_candidate(evidence, event)

    with pytest.raises(MotivationEvidenceConflict, match="event identity was reused"):
        system.apply_evidence(evidence, event, candidate=candidate)
    assert system.snapshot().authority_digest == before.authority_digest


def test_conflicting_motivations_coexist_without_winner_selection() -> None:
    system = MotivationSystem()
    target = ref(R13ReferenceKind.VALUE, "value:conflict")
    desire = make_evidence(
        ref(R13ReferenceKind.EXPERIENCE, "experience:approach"),
        target=target,
        kind=MotivationKind.DESIRE,
    )
    aversion = make_evidence(
        ref(R13ReferenceKind.EXPERIENCE, "experience:avoid"),
        target=target,
        kind=MotivationKind.AVERSION,
    )
    first = system.apply_evidence(desire, source_event(desire, 1))
    second = system.apply_evidence(aversion, source_event(aversion, 2))

    assert first.motivation_id != second.motivation_id
    assert {item.kind for item in system.records} == {
        MotivationKind.DESIRE,
        MotivationKind.AVERSION,
    }
    assert all(item.lifecycle is MotivationLifecycle.ACTIVE for item in system.records)


def test_decay_review_and_satiation_are_deterministic_and_event_bound() -> None:
    def run() -> MotivationSystem:
        current = MotivationSystem()
        evidence = make_evidence(salience=1.0, confidence=1.0)
        created = current.apply_evidence(evidence, source_event(evidence, 1))
        motivation_ref = ref(R13ReferenceKind.MOTIVATION, created.motivation_id)
        decay_event = make_event(
            (motivation_ref,),
            sequence=2,
            recorded_at=NOW + timedelta(days=1),
        )
        decayed = current.decay(decay_event, elapsed_seconds=24 * 60 * 60)[0]
        assert decayed.lifecycle is MotivationLifecycle.DORMANT
        assert decayed.strength < created.strength

        review_event = make_event(
            (motivation_ref,),
            sequence=3,
            recorded_at=NOW + timedelta(days=2),
        )
        reviewed = current.review(created.motivation_id, review_event)
        assert reviewed.lifecycle is MotivationLifecycle.ACTIVE

        satiation_event = make_event(
            (motivation_ref,),
            sequence=4,
            recorded_at=NOW + timedelta(days=3),
        )
        satiated = current.satiate(created.motivation_id, satiation_event)
        assert satiated.lifecycle is MotivationLifecycle.SATIATED
        assert satiated.satiation == 0.25

        reactivation_event = make_event(
            (motivation_ref,),
            sequence=5,
            recorded_at=NOW + timedelta(days=4),
        )
        reactivated = current.review(created.motivation_id, reactivation_event)
        assert reactivated.lifecycle is MotivationLifecycle.ACTIVE
        assert reactivated.satiation == 0.0
        return current

    first = run()
    second = run()
    assert first.snapshot().authority_digest == second.snapshot().authority_digest


@pytest.mark.parametrize(
    "elapsed",
    [-1.0, math.nan, math.inf, -math.inf, True, MOTIVATION_MAX_ELAPSED_SECONDS + 1],
)
def test_invalid_elapsed_time_fails_without_partial_mutation(elapsed: object) -> None:
    system = MotivationSystem()
    evidence = make_evidence()
    record = system.apply_evidence(evidence, source_event(evidence, 1))
    event = make_event(
        (ref(R13ReferenceKind.MOTIVATION, record.motivation_id),),
        sequence=2,
    )
    before = system.snapshot()
    with pytest.raises((TypeError, ValueError)):
        system.decay(event, elapsed_seconds=elapsed)  # type: ignore[arg-type]
    assert system.snapshot().authority_digest == before.authority_digest


def test_future_source_and_regressed_clock_fail_atomically() -> None:
    system = MotivationSystem()
    future = make_evidence(
        ref(R13ReferenceKind.EXPERIENCE, "experience:future"),
        observed_at=NOW + timedelta(days=2),
    )
    before = system.snapshot()
    with pytest.raises(MotivationDomainError, match="timestamp is in the future"):
        system.apply_evidence(
            future,
            source_event(
                future,
                1,
                recorded_at=NOW + timedelta(days=1),
            ),
        )
    assert system.snapshot().authority_digest == before.authority_digest

    first = make_evidence(ref(R13ReferenceKind.EXPERIENCE, "experience:first"))
    system.apply_evidence(first, source_event(first, 1, recorded_at=NOW + timedelta(days=2)))
    second = make_evidence(ref(R13ReferenceKind.EXPERIENCE, "experience:second"))
    before_regressed = system.snapshot()
    with pytest.raises(MotivationDomainError, match="time regressed"):
        system.apply_evidence(second, source_event(second, 2, recorded_at=NOW))
    assert system.snapshot().authority_digest == before_regressed.authority_digest


def test_revision_overflow_fails_closed_without_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    system = MotivationSystem()
    first = make_evidence()
    record = system.apply_evidence(first, source_event(first, 1))
    before = system.snapshot()
    monkeypatch.setattr(motivation_system_module, "R13_MAX_REVISION", record.revision)
    later = make_evidence(ref(R13ReferenceKind.EXPERIENCE, "experience:later"))
    with pytest.raises(MotivationCapacityExceeded, match="revision bound"):
        system.apply_evidence(later, source_event(later, 2))
    monkeypatch.undo()
    assert system.snapshot().authority_digest == before.authority_digest


def test_exact_satiation_retry_is_idempotent_at_revision_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = MotivationSystem()
    evidence = make_evidence()
    record = system.apply_evidence(evidence, source_event(evidence, 1))
    motivation_ref = ref(R13ReferenceKind.MOTIVATION, record.motivation_id)
    event = make_event((motivation_ref,), sequence=2)

    monkeypatch.setattr(motivation_system_module, "R13_MAX_REVISION", 1)
    satiated = system.satiate(record.motivation_id, event)
    assert satiated.revision == 1
    before_retry = system.snapshot()
    assert system.satiate(record.motivation_id, event) == satiated
    assert system.snapshot().authority_digest == before_retry.authority_digest

    changed_event = replace(event, recorded_at=event.recorded_at + timedelta(seconds=1))
    with pytest.raises(MotivationEvidenceConflict, match="event identity was reused"):
        system.satiate(record.motivation_id, changed_event)
    assert system.snapshot().authority_digest == before_retry.authority_digest


def test_record_evidence_and_global_ledger_capacity_fail_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = MotivationSystem()
    first = make_evidence()
    created = system.apply_evidence(first, source_event(first, 1))
    before = system.snapshot()
    monkeypatch.setattr(motivation_system_module, "MOTIVATION_MAX_EVIDENCE_PER_RECORD", 1)
    later = make_evidence(ref(R13ReferenceKind.EXPERIENCE, "experience:later"))
    with pytest.raises(MotivationCapacityExceeded, match="reference bound"):
        system.apply_evidence(later, source_event(later, 2))
    monkeypatch.undo()
    assert system.snapshot().authority_digest == before.authority_digest

    monkeypatch.setattr(motivation_system_module, "MOTIVATION_MAX_EVIDENCE_LEDGER", 1)
    other = make_evidence(
        ref(R13ReferenceKind.EXPERIENCE, "experience:other"),
        target=ref(R13ReferenceKind.VALUE, "value:other"),
    )
    with pytest.raises(MotivationCapacityExceeded, match="identity ledger is full"):
        system.apply_evidence(other, source_event(other, 2))
    monkeypatch.undo()
    assert system.snapshot().authority_digest == before.authority_digest
    assert system.get(created.motivation_id) == created


def test_record_capacity_event_budget_and_candidate_capacity_fail_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = MotivationSystem()
    first = make_evidence()
    system.apply_evidence(first, source_event(first, 1))
    before = system.snapshot()
    monkeypatch.setattr(motivation_system_module, "MOTIVATION_MAX_RECORDS", 1)
    other = make_evidence(
        ref(R13ReferenceKind.EXPERIENCE, "experience:other"),
        target=ref(R13ReferenceKind.VALUE, "value:other"),
    )
    with pytest.raises(MotivationCapacityExceeded, match="record capacity"):
        system.apply_evidence(other, source_event(other, 2))
    monkeypatch.undo()
    assert system.snapshot().authority_digest == before.authority_digest

    event = source_event(first, 2, event_id="event:candidate")
    candidate = make_candidate(first, event)
    monkeypatch.setattr(motivation_system_module, "MOTIVATION_MAX_CANDIDATES", 0)
    before_candidate = system.snapshot()
    with pytest.raises(MotivationCapacityExceeded, match="candidate ledger is full"):
        system.apply_evidence(first, event, candidate=candidate)
    monkeypatch.undo()
    assert system.snapshot().authority_digest == before_candidate.authority_digest


def test_per_event_evidence_and_event_id_reuse_fail_closed() -> None:
    system = MotivationSystem()
    first = make_evidence()
    event = source_event(first, 1)
    system.apply_evidence(first, event)
    second = make_evidence(ref(R13ReferenceKind.EXPERIENCE, "experience:second"))
    before = system.snapshot()
    with pytest.raises(MotivationEvidenceConflict, match="event identity was reused"):
        system.apply_evidence(second, source_event(second, 1))
    assert system.snapshot().authority_digest == before.authority_digest


def test_goal_proposal_is_bounded_deterministic_proposed_and_admission_free() -> None:
    system = MotivationSystem()
    evidence = make_evidence(salience=1.0, confidence=1.0)
    event = source_event(evidence, 1)
    record = system.apply_evidence(evidence, event)

    proposal = system.propose_goal(record.motivation_id)
    assert proposal is not None
    assert proposal.lifecycle.value == "proposed"
    assert proposal.subject_admission is None
    assert proposal.target == ref(R13ReferenceKind.MOTIVATION, record.motivation_id)
    assert proposal.evidence_refs == tuple(
        sorted(
            (proposal.target, record.source_evidence),
            key=lambda item: (item.reference, item.kind.value),
        )
    )
    assert proposal.origin_refs == (proposal.target,)
    assert proposal.revision_history[0].event_id == event.event_id
    assert system.propose_goal(record.motivation_id) == proposal

    later = make_evidence(
        ref(R13ReferenceKind.EXPERIENCE, "experience:later"),
        target=record.target,
        salience=1.0,
        confidence=1.0,
    )
    system.apply_evidence(later, source_event(later, 2))
    assert system.propose_goal(record.motivation_id) == proposal


def test_high_salience_evidence_stays_within_per_event_record_and_goal_budgets() -> None:
    system = MotivationSystem()
    evidence = make_evidence(salience=1.0, confidence=1.0, persistence=1.0)
    record = system.apply_evidence(evidence, source_event(evidence, 1))
    assert record.strength <= 0.30
    assert len(system.records) == 1
    proposal = system.propose_goal(record.motivation_id)
    assert proposal is not None
    assert len((proposal,)) <= motivation_system_module.MOTIVATION_MAX_GOALS_PER_EVENT
    assert len((proposal,)) <= motivation_system_module.MOTIVATION_MAX_GOALS_PER_MOTIVATION


def test_snapshot_is_exact_bounded_and_rejects_byte_overflow_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = MotivationSystem()
    evidence = make_evidence()
    system.apply_evidence(evidence, source_event(evidence, 1))
    before = system.snapshot()
    assert before.serialized_bytes <= MOTIVATION_SYSTEM_MAX_SERIALIZED_BYTES
    assert before.authority_digest == system.export().authority_digest

    second = make_evidence(ref(R13ReferenceKind.EXPERIENCE, "experience:second"))
    with monkeypatch.context() as patch:
        patch.setattr(motivation_system_module, "MOTIVATION_SYSTEM_MAX_SERIALIZED_BYTES", 1)
        with pytest.raises(MotivationCapacityExceeded, match="byte bound"):
            system.apply_evidence(second, source_event(second, 2))
    assert system.snapshot().authority_digest == before.authority_digest


def test_snapshot_rejects_evidence_provenance_not_bound_by_its_receipt() -> None:
    system = MotivationSystem()
    evidence = make_evidence()
    system.apply_evidence(evidence, source_event(evidence, 1))
    snapshot = system.snapshot()
    entry = snapshot.evidence_ledger[0]
    altered = replace(
        entry,
        evidence=replace(
            entry.evidence,
            origin_refs=(ref(R13ReferenceKind.SYSTEM, "system:laundered"),),
        ),
    )

    with pytest.raises(ValueError, match="does not bind exact evidence provenance"):
        type(snapshot)(
            records=snapshot.records,
            evidence_ledger=(altered,),
            candidates=snapshot.candidates,
            event_receipts=snapshot.event_receipts,
        )

    receipt = snapshot.event_receipts[0]
    with pytest.raises(ValueError, match="evidence receipt digest is inconsistent"):
        type(snapshot)(
            records=snapshot.records,
            evidence_ledger=snapshot.evidence_ledger,
            candidates=snapshot.candidates,
            event_receipts=(replace(receipt, input_digest="0" * 64),),
        )


def test_snapshot_rejects_orphan_evidence_receipts_and_time_mismatch() -> None:
    system = MotivationSystem()
    evidence = make_evidence()
    system.apply_evidence(evidence, source_event(evidence, 1))
    snapshot = system.snapshot()
    receipt = snapshot.event_receipts[0]

    orphan = replace(
        receipt,
        event_id="event:orphan",
        event_sequence=2,
        recorded_at=receipt.recorded_at + timedelta(seconds=1),
    )
    with pytest.raises(ValueError, match="one exact accepted input"):
        type(snapshot)(
            records=snapshot.records,
            evidence_ledger=snapshot.evidence_ledger,
            candidates=snapshot.candidates,
            event_receipts=(receipt, orphan),
        )

    time_tampered = replace(
        receipt,
        recorded_at=receipt.recorded_at + timedelta(seconds=1),
    )
    with pytest.raises(ValueError, match="exact event receipt"):
        type(snapshot)(
            records=snapshot.records,
            evidence_ledger=snapshot.evidence_ledger,
            candidates=snapshot.candidates,
            event_receipts=(time_tampered,),
        )


def test_motivation_history_compacts_without_losing_evidence_idempotency() -> None:
    system = MotivationSystem()
    first = make_evidence(ref(R13ReferenceKind.EXPERIENCE, "experience:00"))
    record = system.apply_evidence(first, source_event(first, 1))
    for sequence in range(2, 12):
        evidence = make_evidence(
            ref(R13ReferenceKind.EXPERIENCE, f"experience:{sequence:02d}"),
            target=record.target,
        )
        record = system.apply_evidence(evidence, source_event(evidence, sequence))
    assert record.history_anchor is not None
    assert len(record.revision_history) == 8
    assert len(record.evidence_refs) == 11
    proposal = system.propose_goal(record.motivation_id)
    assert proposal is not None
    assert proposal.revision_history[0].event_id == "event:1"
    before = system.snapshot()
    assert system.apply_evidence(first, source_event(first, 12)) == record
    assert system.snapshot().authority_digest == before.authority_digest


def test_system_api_import_does_not_load_models_runtime_or_timer_modules() -> None:
    import ast
    from pathlib import Path

    source_path = Path(motivation_system_module.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not imported.intersection({"torch", "transformers", "threading", "asyncio", "sched"})
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import suzka.motivation; "
                "assert 'suzka.runtime' not in sys.modules; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
