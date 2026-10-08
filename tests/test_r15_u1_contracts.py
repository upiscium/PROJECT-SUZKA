"""U1-only adversarial evidence: shape is not source or subject authority."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
import json

import pytest

from suzka.r15.common import (
    MAX_COUNTER,
    MAX_RELATIONSHIPS,
    MAX_SUPPORT,
    SCORE_SCALE,
    Accessibility,
    ClaimedOrigin,
    Concept,
    EventRef,
    Inertia,
    Interpretation,
    InterpretationStatus,
    RevisionProof,
    RevisionReason,
    SourceDisposition,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
    UntrustedIngress,
    asserted_intrinsic_disposition,
    checksum,
    source_disposition,
)
from suzka.r15.contracts import (
    CapabilityHypothesis,
    NarrativeClaim,
    NarrativeClaimKind,
    NarrativeEpisode,
    NarrativeSelfState,
    RelationshipAxis,
    RelationshipAxisKind,
    RelationshipRecord,
    RelationshipState,
    SelfClaim,
    SelfClaimKind,
    SelfModelState,
    VerifiedCompetence,
    RELATIONSHIP_DOMAIN,
)
from suzka.r15.stance import (
    ActorCausation,
    FeltAssociation,
    InteractionStance,
    context_projection_checksum,
    emotion_projection_checksum,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)
DIGEST = "f" * 64


def event(sequence: int = 1) -> EventRef:
    return EventRef("event-1", sequence, NOW)


def source(
    kind: SourceKind = SourceKind.EXPERIENCE,
    index: int = 0,
    *,
    lifecycle: SourceLifecycle = SourceLifecycle.ACTIVE,
    origin: ClaimedOrigin = ClaimedOrigin.SUBJECT_OBSERVATION,
) -> SourceWitness:
    return SourceWitness(
        kind, f"source-{index}", 0, DIGEST, DIGEST, lifecycle, origin,
        event() if kind in (SourceKind.EXPERIENCE, SourceKind.EMOTION) else None,
    )


def claim(
    index: int = 1,
    *,
    status: InterpretationStatus = InterpretationStatus.PRESENT,
    witnesses: tuple[SourceWitness, ...] | None = None,
) -> Interpretation:
    return Interpretation(
        f"{index:064x}",
        status,
        None if status in (InterpretationStatus.UNKNOWN, InterpretationStatus.CONTRADICTORY) else SCORE_SCALE,
        None if status in (InterpretationStatus.UNKNOWN, InterpretationStatus.CONTRADICTORY) else SCORE_SCALE,
        () if status is InterpretationStatus.UNKNOWN else (witnesses if witnesses is not None else (source(index=index),)),
        (source(index=index + 100),) if status in (InterpretationStatus.CONTESTED, InterpretationStatus.CONTRADICTORY) else (),
        Accessibility.CURRENT,
        Inertia.REVISABLE,
    )


def relationship(index: int = 1) -> RelationshipRecord:
    return RelationshipRecord(
        f"{index:064x}", f"interlocutor-{index}", "unproved-same-person",
        0,
        tuple(
            RelationshipAxis(kind, claim(1000 + index * 6 + offset, status=InterpretationStatus.UNKNOWN))
            for offset, kind in enumerate(sorted(RelationshipAxisKind, key=lambda item: item.value))
        ),
    )


def test_r15_closed_roots_roundtrip_is_canonical_and_independent() -> None:
    rel = RelationshipState(1, 0, (relationship(),), None, ())
    narrative = NarrativeSelfState(1, 0, (), (), None, ())
    self_model = SelfModelState(1, 0, (), (), None, ())
    for root, root_type in (
        (rel, RelationshipState), (narrative, NarrativeSelfState), (self_model, SelfModelState)
    ):
        data = root.canonical_bytes()
        assert root_type.from_json(data).canonical_bytes() == data
        assert root_type.from_json(data).canonical_bytes() == data
        with pytest.raises(FrozenInstanceError):
            root.revision = 99  # type: ignore[misc]
    assert "relationship_state" not in narrative.canonical_value()
    assert "self_model_state" not in rel.canonical_value()
    assert "narrative_self_state" not in self_model.canonical_value()


def test_r15_json_requires_closed_exact_canonical_shape_and_known_version() -> None:
    root = RelationshipState(1, 0, (), None, ())
    good = root.canonical_bytes()
    for bad in (
        b"{\"revision\":0,\"revision\":1}",
        b"[" * 40 + b"0" + b"]" * 40,
        good + b" ",
        good.replace(b"1", b"NaN", 1),
        good.replace(b"1", b"Infinity", 1),
        good.replace(b"\"records\":[]", b"\"records\":{},\"raw_prompt\":\"secret\""),
    ):
        with pytest.raises((TypeError, ValueError)):
            RelationshipState.from_json(bad)
    for field, value in (
        ("schema_version", 2),
        ("schema_version", True),
        ("revision", True),
        ("revision", 2**31),
        ("records", {}),
        ("raw_transcript", "private"),
    ):
        raw = root.canonical_value()
        raw[field] = value
        raw["state_digest"] = checksum(RELATIONSHIP_DOMAIN, {key: item for key, item in raw.items() if key != "state_digest"})
        with pytest.raises((TypeError, ValueError)):
            RelationshipState.from_value(raw)
    with pytest.raises(ValueError):
        RelationshipState.from_json(json.dumps(root.canonical_value(), indent=2).encode())


def test_r15_missingness_conflict_and_fixed_point_are_not_implicit_confidence() -> None:
    assert claim(status=InterpretationStatus.UNKNOWN).confidence is None
    assert claim(status=InterpretationStatus.CONTESTED).counter
    assert claim(status=InterpretationStatus.CONTRADICTORY).estimate is None
    with pytest.raises((ValueError, TypeError)):
        replace(claim(), confidence=None)
    with pytest.raises((ValueError, TypeError)):
        replace(claim(), confidence=True)
    with pytest.raises(ValueError):
        replace(claim(), confidence=SCORE_SCALE + 1)
    with pytest.raises(ValueError):
        replace(claim(), support=(source(index=1),) * (MAX_SUPPORT + 1))
    with pytest.raises(ValueError):
        replace(claim(), counter=(source(index=1),) * (MAX_COUNTER + 1))
    with pytest.raises(ValueError):
        replace(claim(status=InterpretationStatus.CONTESTED), counter=(source(index=1),))


def test_r15_bounded_identifier_evidence_revision_and_ordering_reject() -> None:
    with pytest.raises(ValueError):
        source(index=1).__class__(SourceKind.EXPERIENCE, "x" * 129, 0, DIGEST, DIGEST,
                                  SourceLifecycle.ACTIVE, ClaimedOrigin.SUBJECT_OBSERVATION, event())
    with pytest.raises(ValueError):
        replace(source(), source_digest="not-a-digest")
    with pytest.raises(ValueError):
        replace(source(), revision=True)
    with pytest.raises(ValueError):
        replace(event(), event_sequence=True)
    with pytest.raises(ValueError):
        replace(event(), occurred_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        RelationshipState(1, 0, (relationship(2), relationship(1)), None, ())
    with pytest.raises(ValueError):
        RelationshipState(1, 0, tuple(relationship(i) for i in range(MAX_RELATIONSHIPS + 1)), None, ())
    with pytest.raises(ValueError):
        replace(relationship(), axes=relationship().axes[:-1])


def test_r15_claimed_identity_is_not_person_proof_or_alias_merge() -> None:
    left, right = relationship(1), relationship(2)
    assert left.claimed_identity_key == right.claimed_identity_key
    root = RelationshipState(1, 0, (left, right), None, ())
    assert len(RelationshipState.from_json(root.canonical_bytes()).records) == 2
    binding = source(SourceKind.BINDING, lifecycle=SourceLifecycle.DECLARATIVE, origin=ClaimedOrigin.DECLARATIVE)
    assert source_disposition(Concept.CLAIMED_IDENTITY, binding) is SourceDisposition.REQUIRES_TRUSTED_ROOT
    assert source_disposition(Concept.VERIFIED_PERSON, binding) is SourceDisposition.UNAVAILABLE
    assert source_disposition(Concept.RELATIONSHIP, binding) is SourceDisposition.UNAVAILABLE


@pytest.mark.parametrize("ingress", list(UntrustedIngress))
@pytest.mark.parametrize("concept", (Concept.RELATIONSHIP, Concept.SELF_HYPOTHESIS, Concept.VERIFIED_COMPETENCE))
def test_r15_d9_every_planned_ingress_claim_fails_closed(ingress: UntrustedIngress, concept: Concept) -> None:
    assert asserted_intrinsic_disposition(concept, ingress) is SourceDisposition.UNAVAILABLE


def test_r15_fake_provenance_subject_admission_and_untrusted_text_cannot_adopt() -> None:
    for kind in (SourceKind.OPERATOR_CLAIM, SourceKind.MODEL_TEXT, SourceKind.WEB_TEXT, SourceKind.VERIFIED_OUTCOME):
        fake = source(kind)
        for concept in (Concept.RELATIONSHIP, Concept.SELF_HYPOTHESIS, Concept.VERIFIED_COMPETENCE):
            assert source_disposition(concept, fake) is SourceDisposition.UNAVAILABLE
        with pytest.raises(ValueError):
            RelationshipAxis(RelationshipAxisKind.TRUST, claim(witnesses=(fake,)))
    forged_subject = source(index=789)
    assert source_disposition(Concept.RELATIONSHIP, forged_subject) is SourceDisposition.REQUIRES_TRUSTED_ROOT
    # Even an attacker-computed *valid checksum* is not source authentication.
    assert "admitted" not in {state.value for state in SourceDisposition}
    with pytest.raises(ValueError):
        SelfClaim(SelfClaimKind.POSSIBLE_TRAIT, claim(witnesses=(replace(forged_subject, origin=ClaimedOrigin.OPERATOR_ASSERTION),)))


def test_r15_uncommitted_retracted_superseded_experience_has_no_current_proof() -> None:
    for state in (SourceLifecycle.SUPERSEDED, SourceLifecycle.RETRACTED, SourceLifecycle.UNAVAILABLE):
        stale = source(lifecycle=state)
        assert source_disposition(Concept.NARRATIVE_EPISODE, stale) is SourceDisposition.UNAVAILABLE
        with pytest.raises(ValueError):
            NarrativeEpisode(DIGEST, stale, claim(witnesses=(stale,)), ())
    uncommitted = source()
    assert source_disposition(Concept.NARRATIVE_EPISODE, uncommitted) is SourceDisposition.REQUIRES_TRUSTED_ROOT
    # The U1 tuple does not contain a trusted R12 commit receipt. A same-event
    # record construction or claimed 'active' lifecycle cannot bypass finalization.
    assert not hasattr(uncommitted, "admitted")
    with pytest.raises(ValueError):
        NarrativeEpisode(DIGEST, source(index=123), claim(witnesses=(source(index=456),)), ())


def test_r15_narrative_links_conflict_and_self_hypothesis_do_not_mutate_sources() -> None:
    observation = source(index=42)
    episode = NarrativeEpisode(DIGEST, observation, claim(witnesses=(observation,)), ())
    contested = claim(41, status=InterpretationStatus.CONTESTED)
    narrative = NarrativeSelfState(1, 0, (episode,), (NarrativeClaim(NarrativeClaimKind.CONTRADICTION, contested, (DIGEST,)),), None, ())
    self_model = SelfModelState(1, 0, (SelfClaim(SelfClaimKind.LIMITATION, claim(55)),), (), None, ())
    assert NarrativeSelfState.from_json(narrative.canonical_bytes()).claims[0].interpretation.counter
    assert SelfModelState.from_json(self_model.canonical_bytes()).claims[0].interpretation.status is InterpretationStatus.PRESENT
    assert observation == source(index=42)
    with pytest.raises(ValueError):
        NarrativeSelfState(1, 0, (episode,), (NarrativeClaim(NarrativeClaimKind.ROLE, claim(2), ("a" * 64,)),), None, ())


def test_r15_verified_competence_never_from_attempt_handler_or_decision() -> None:
    attempt = CapabilityHypothesis("python-task", claim(), VerifiedCompetence.UNKNOWN)
    assert attempt.attempted.confidence == SCORE_SCALE
    assert attempt.verified_competence is VerifiedCompetence.UNKNOWN
    with pytest.raises((TypeError, ValueError)):
        replace(attempt, verified_competence="verified")
    assert source_disposition(Concept.VERIFIED_COMPETENCE, source(SourceKind.VERIFIED_OUTCOME)) is SourceDisposition.UNAVAILABLE


def test_r15_revision_suffix_reason_and_technical_restore_are_distinct() -> None:
    first = RevisionProof(1, event(), DIGEST, DIGEST, DIGEST, (), (), RevisionReason.INITIAL_INTERPRETATION, None)
    assert first.reason is not RevisionReason.ACCESSIBILITY_REVIEW
    valid = RelationshipState(1, 1, (), None, (first,))
    assert RelationshipState.from_json(valid.canonical_bytes()) == valid
    with pytest.raises(ValueError):
        RelationshipState(1, 2, (), None, (first,))
    with pytest.raises(ValueError):
        RelationshipState(1, 1, (), None, (replace(first, event=EventRef("different", 0, NOW)),))
    assert RevisionReason.QUALIFIED_REINTERPRETATION.value != "rollback"
    assert Accessibility.LATENT.value != SourceLifecycle.RETRACTED.value


def test_r15_stance_is_current_event_only_not_actor_blame_or_durable_trust() -> None:
    current = event()
    value = (0.0).hex()
    arousal = (1.0).hex()
    optimal = (2.0).hex()
    emotion = replace(
        source(SourceKind.EMOTION),
        source_digest=emotion_projection_checksum(current, value, arousal, optimal),
    )
    context = replace(
        source(SourceKind.CONTEXT),
        event=current,
        source_digest=context_projection_checksum(current, "source-0", 0, ("interlocutor-1",)),
    )
    stance = InteractionStance(
        current, context, emotion, ("interlocutor-1",), "interlocutor-1", value, arousal, optimal, None,
        FeltAssociation.TENTATIVE_INTERACTION, SCORE_SCALE, ActorCausation.UNKNOWN,
    )
    assert InteractionStance.from_value(stance.canonical_value()).canonical_bytes() == stance.canonical_bytes()
    assert stance.actor_causation is ActorCausation.UNKNOWN
    assert "trust" not in stance.canonical_value()
    assert "anger" not in stance.canonical_value()
    with pytest.raises(ValueError):
        replace(stance, emotion=replace(emotion, event=EventRef("event-1", 2, NOW + timedelta(seconds=1))))
    with pytest.raises(ValueError):
        replace(stance, valence_hex=(-1.0).hex())
    with pytest.raises(ValueError):
        replace(stance, association=FeltAssociation.UNKNOWN)
    with pytest.raises((ValueError, TypeError)):
        replace(stance, actor_causation="operator_says_actor_caused_it")
    with pytest.raises(ValueError):
        replace(stance, valence_hex="nan")
    with pytest.raises(ValueError):
        replace(stance, interlocutor_key="not-in-context")
    with pytest.raises(ValueError):
        replace(stance, context=replace(context, event=None))
    with pytest.raises(ValueError):
        replace(stance, context_participant_refs=("interlocutor-2",))
    with pytest.raises(FrozenInstanceError):
        stance.uncertainty = None  # type: ignore[misc]
