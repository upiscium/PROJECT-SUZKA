"""Full-bound R15 U1 capacity proof over actual closed canonical serializers."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from suzka.r15.bounds import (
    AGENT_STATE_HARD_CAP_BYTES,
    FULL_FUTURE_RESERVE_BYTES,
    R15_ROOT_NAMES,
    V9_BASE_MAX_BYTES,
    V9_WITH_RESERVE_BYTES,
    derive_r15_schema_budget,
    maximum_legal_r15_roots,
    require_r15_capacity,
)
from suzka.r15.common import (
    MAX_CAPABILITIES,
    MAX_COUNTER,
    MAX_EPISODES,
    MAX_LINKS,
    MAX_NARRATIVE_CLAIMS,
    MAX_RELATIONSHIPS,
    MAX_REVISIONS,
    MAX_SELF_CLAIMS,
    MAX_SUPPORT,
    ClaimedOrigin,
    Concept,
    EventRef,
    Accessibility,
    Inertia,
    InterpretationStatus,
    RevisionReason,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
    canonical_json,
    encode,
    source_disposition,
)
from suzka.r15.contracts import (
    NarrativeClaimKind,
    NarrativeSelfState,
    PropositionPolarity,
    RelationshipState,
    SemanticCode,
    SemanticReferentKind,
    SelfClaimKind,
    SelfModelState,
    ClaimMeaning,
)


def test_r15_full_legal_fixture_fills_every_window_and_roundtrips() -> None:
    relationship, narrative, self_model = maximum_legal_r15_roots()
    assert len(relationship.records) == MAX_RELATIONSHIPS
    assert len(narrative.episodes) == MAX_EPISODES
    assert len(narrative.claims) == MAX_NARRATIVE_CLAIMS
    assert len(self_model.claims) == MAX_SELF_CLAIMS
    assert len(self_model.capabilities) == MAX_CAPABILITIES
    for root, root_type in (
        (relationship, RelationshipState), (narrative, NarrativeSelfState), (self_model, SelfModelState),
    ):
        assert len(root.revision_history) == MAX_REVISIONS
        assert all(len(proof.support_digests) == MAX_SUPPORT for proof in root.revision_history)
        assert all(len(proof.counter_digests) == MAX_COUNTER for proof in root.revision_history)
        assert root_type.from_json(root.canonical_bytes()).canonical_bytes() == root.canonical_bytes()
    for record in relationship.records:
        assert all(len(axis.interpretation.support) == MAX_SUPPORT for axis in record.axes)
        assert all(len(axis.interpretation.counter) == MAX_COUNTER for axis in record.axes)
    assert all(len(episode.links) == MAX_LINKS for episode in narrative.episodes)
    assert all(len(claim.episode_ids) == MAX_LINKS for claim in narrative.claims)
    assert all(len(cap.attempted.counter) == MAX_COUNTER for cap in self_model.capabilities)


def test_r15_v10_projection_matches_integrated_v9_projector_and_keeps_full_reserve() -> None:
    from suzka.runtime.agent_state import (
        AGENT_STATE_FUTURE_STATE_RESERVE_BYTES,
        AGENT_STATE_MAX_SERIALIZED_BYTES,
        AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES,
        AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES,
        AgentStateSnapshotV9,
        project_agent_state_schema_max_bytes,
    )

    budget = require_r15_capacity()
    assert V9_BASE_MAX_BYTES == AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES == 113_767_203
    assert V9_WITH_RESERVE_BYTES == AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES == 130_544_419
    assert AGENT_STATE_MAX_SERIALIZED_BYTES == AGENT_STATE_HARD_CAP_BYTES == 134_217_728
    assert AGENT_STATE_FUTURE_STATE_RESERVE_BYTES == FULL_FUTURE_RESERVE_BYTES == 16_777_216
    assert not set(R15_ROOT_NAMES).intersection(AgentStateSnapshotV9.model_fields)
    assert budget.root_overhead_bytes == sum(len(canonical_json(name)) + 2 for name in R15_ROOT_NAMES)
    assert budget.v9_schema_version_delta_bytes == 1
    assert budget.v10_with_reserve_bytes == project_agent_state_schema_max_bytes(
        schema_version=10,
        base_schema_version=9,
        added_field_maxima=dict(zip(R15_ROOT_NAMES, (
            budget.relationship_bytes, budget.narrative_bytes, budget.self_model_bytes
        ), strict=True)),
    )
    assert budget.v10_before_reserve_bytes == 116_949_381
    assert budget.v10_with_reserve_bytes == 133_726_597
    assert budget.remaining_beyond_reserve_bytes == 491_131
    assert AGENT_STATE_HARD_CAP_BYTES - budget.v10_before_reserve_bytes == (
        FULL_FUTURE_RESERVE_BYTES + budget.remaining_beyond_reserve_bytes
    )


def test_r15_section_envelopes_exhaust_actual_maximum_legal_values() -> None:
    budget = derive_r15_schema_budget()
    relationship, narrative, self_model = maximum_legal_r15_roots()
    assert tuple(len(root.canonical_bytes()) for root in (relationship, narrative, self_model)) == (
        budget.relationship_bytes, budget.narrative_bytes, budget.self_model_bytes
    ) == (2_159_153, 678_461, 344_497)
    assert all(len(item.interlocutor_key) == 128 for item in relationship.records)
    assert all(len(item.claimed_identity_key or "") == 128 for item in relationship.records)
    assert all(len(witness.reference) == 128 for record in relationship.records
               for axis in record.axes for witness in axis.interpretation.support + axis.interpretation.counter)
    assert all(len(proof.event.event_id) == 128 for root in (relationship, narrative, self_model)
               for proof in root.revision_history)
    assert all(proof.event.occurred_at == datetime.max.replace(tzinfo=UTC)
               for root in (relationship, narrative, self_model) for proof in root.revision_history)
    assert all(len(item.meaning.referent) == 64 for item in narrative.claims)
    assert all(len(item.meaning.referent) == 128 for item in self_model.claims)
    assert all(item.interpretation.statement_id == item.meaning.statement_id(
        domain="narrative", kind=item.kind.value, episode_ids=item.episode_ids
    ) for item in narrative.claims)
    assert all(item.interpretation.statement_id == item.meaning.statement_id(
        domain="self_model", kind=item.kind.value
    ) for item in self_model.claims)


def test_r15_capacity_is_not_a_runtime_clipper_and_one_over_fails_closed() -> None:
    relationship, narrative, self_model = maximum_legal_r15_roots()
    # The legal full-width payload is not resized, normalized away or silently
    # evicted when an additional record, revision, link or evidence arrives.
    with pytest.raises(ValueError):
        RelationshipState(1, relationship.revision,
                          relationship.records + (replace(relationship.records[-1], relationship_id="f" * 64, interlocutor_key="z" * 128),),
                          relationship.history_anchor, relationship.revision_history)
    with pytest.raises(ValueError):
        replace(narrative, episodes=narrative.episodes + (narrative.episodes[0],))
    with pytest.raises(ValueError):
        replace(self_model, capabilities=self_model.capabilities + (self_model.capabilities[0],))
    with pytest.raises(ValueError):
        replace(relationship, revision_history=relationship.revision_history + (relationship.revision_history[-1],))
    with pytest.raises(ValueError):
        replace(relationship.records[0].axes[0].interpretation,
                counter=relationship.records[0].axes[0].interpretation.counter + (relationship.records[0].axes[0].interpretation.counter[0],))
    assert require_r15_capacity() == derive_r15_schema_budget()


def test_r15_source_maximum_selected_from_all_admissible_canonical_shapes() -> None:
    # The only alternative current NarrativeClaim input is an adopted Belief;
    # with full event metadata its kind is shorter than committed Experience.
    event = EventRef("e" * 128, 2**63 - 1, datetime.max.replace(tzinfo=UTC))
    experience = SourceWitness(SourceKind.EXPERIENCE, "x" * 128, 2**31 - 1,
                               "f" * 64, "f" * 64, SourceLifecycle.ACTIVE,
                               ClaimedOrigin.SUBJECT_OBSERVATION, event)
    belief = SourceWitness(SourceKind.BELIEF, "x" * 128, 2**31 - 1,
                           "f" * 64, "f" * 64, SourceLifecycle.ADOPTED,
                           ClaimedOrigin.SUBJECT_OBSERVATION, event)
    assert source_disposition(Concept.NARRATIVE_CLAIM, experience).value == "requires_trusted_root"
    assert source_disposition(Concept.NARRATIVE_CLAIM, belief).value == "requires_trusted_root"
    assert len(canonical_json(encode(experience))) > len(canonical_json(encode(belief)))


def test_r15_legal_status_origin_lifecycle_enum_alternatives_do_not_exceed_full_root() -> None:
    relationship, narrative, self_model = maximum_legal_r15_roots()
    max_relationship = len(relationship.canonical_bytes())
    max_narrative = len(narrative.canonical_bytes())
    max_self_model = len(self_model.canonical_bytes())
    first_record = relationship.records[0]
    first_axis = first_record.axes[0]
    baseline = first_axis.interpretation
    for changed in (
        replace(baseline, status=InterpretationStatus.CONTRADICTORY, estimate=None, confidence=None),
        replace(baseline, status=InterpretationStatus.PRESENT, counter=()),
        replace(baseline, status=InterpretationStatus.UNKNOWN, estimate=None, confidence=None, support=(), counter=()),
        replace(baseline, accessibility=Accessibility.LATENT, inertia=Inertia.TENTATIVE),
    ):
        record = replace(first_record, axes=(replace(first_axis, interpretation=changed),) + first_record.axes[1:])
        assert len(replace(relationship, records=(record,) + relationship.records[1:]).canonical_bytes()) <= max_relationship
    for reason in RevisionReason:
        # Changing the last proof leaves its predecessor intact; earlier proof
        # digests would need their successor chain adjusted by a real owner.
        proof = replace(relationship.revision_history[-1], reason=reason)
        alternative = replace(relationship, revision_history=relationship.revision_history[:-1] + (proof,))
        assert len(alternative.canonical_bytes()) <= max_relationship

    first_claim = narrative.claims[0]
    narrative_codes = {
        NarrativeClaimKind.CONTINUITY: (SemanticCode.NARRATIVE_CONTINUITY,),
        NarrativeClaimKind.ROLE: (SemanticCode.NARRATIVE_ROLE_ASSISTANT, SemanticCode.NARRATIVE_ROLE_PARTNER),
        NarrativeClaimKind.REINTERPRETATION: (SemanticCode.NARRATIVE_REINTERPRETATION,),
        NarrativeClaimKind.CONTRADICTION: (SemanticCode.NARRATIVE_CONTRADICTION,),
    }
    for kind, codes in narrative_codes.items():
        for code in codes:
            meaning = replace(first_claim.meaning, code=code)
            alternate_claim = replace(
                first_claim, kind=kind, meaning=meaning,
                interpretation=replace(first_claim.interpretation, statement_id=meaning.statement_id(
                    domain="narrative", kind=kind.value, episode_ids=first_claim.episode_ids
                )),
            )
            ordered = tuple(sorted((alternate_claim,) + narrative.claims[1:], key=lambda item: item.interpretation.statement_id))
            assert len(replace(narrative, claims=ordered).canonical_bytes()) <= max_narrative
    denied_meaning = replace(first_claim.meaning, polarity=PropositionPolarity.DENIES)
    denied_claim = replace(
        first_claim, meaning=denied_meaning,
        interpretation=replace(first_claim.interpretation, statement_id=denied_meaning.statement_id(
            domain="narrative", kind=first_claim.kind.value, episode_ids=first_claim.episode_ids
        )),
    )
    ordered_denial = tuple(sorted((denied_claim,) + narrative.claims[1:], key=lambda item: item.interpretation.statement_id))
    assert len(replace(narrative, claims=ordered_denial).canonical_bytes()) <= max_narrative
    # A committed Experience can be replaced in a *claim* by an adopted Belief;
    # both have maximal event metadata. The Belief spelling is shorter.
    interpretation = first_claim.interpretation
    belief_support = tuple(replace(item, kind=SourceKind.BELIEF, lifecycle=SourceLifecycle.ADOPTED) for item in interpretation.support)
    belief_counter = tuple(replace(item, kind=SourceKind.BELIEF, lifecycle=SourceLifecycle.ADOPTED) for item in interpretation.counter)
    belief_claim = replace(first_claim, interpretation=replace(interpretation, support=belief_support, counter=belief_counter))
    assert len(replace(narrative, claims=(belief_claim,) + narrative.claims[1:]).canonical_bytes()) <= max_narrative
    for lifecycle in (SourceLifecycle.SUPERSEDED, SourceLifecycle.RETRACTED, SourceLifecycle.UNAVAILABLE):
        # The full legal fixture also retains episodes outside every current
        # claim's links; only those can become inactive without invalidating a
        # current proposition's source-owned referent.
        episode = narrative.episodes[-1]
        unknown = replace(episode.interpretation, status=InterpretationStatus.UNKNOWN,
                          estimate=None, confidence=None, support=(), counter=())
        inactive = replace(episode, experience=replace(episode.experience, lifecycle=lifecycle), interpretation=unknown)
        assert len(replace(narrative, episodes=narrative.episodes[:-1] + (inactive,)).canonical_bytes()) < max_narrative
    first_self_claim = self_model.claims[0]
    self_codes = {
        SelfClaimKind.IDENTITY: (SemanticCode.SELF_IDENTITY_SUZKA,),
        SelfClaimKind.ROLE: (SemanticCode.SELF_ROLE_ASSISTANT, SemanticCode.SELF_ROLE_PARTNER),
        SelfClaimKind.POSSIBLE_TRAIT: (SemanticCode.SELF_TRAIT_CAUTIOUS, SemanticCode.SELF_TRAIT_PATIENT),
        SelfClaimKind.LIMITATION: (
            SemanticCode.SELF_LIMITATION_UNVERIFIED_COMPETENCE,
            SemanticCode.SELF_LIMITATION_DIRECT_ACTION,
        ),
        SelfClaimKind.EPISTEMIC_UNKNOWN: (SemanticCode.SELF_EPISTEMIC_TASK_UNVERIFIED,),
    }
    for kind, codes in self_codes.items():
        for code in codes:
            meaning = ClaimMeaning(
                code,
                SemanticReferentKind.TASK_CLASS if kind is SelfClaimKind.EPISTEMIC_UNKNOWN else SemanticReferentKind.EXPERIENCE,
                self_model.capabilities[0].task_class_key if kind is SelfClaimKind.EPISTEMIC_UNKNOWN else first_self_claim.meaning.referent,
                PropositionPolarity.AFFIRMS,
            )
            interpretation = first_self_claim.interpretation
            if kind is SelfClaimKind.EPISTEMIC_UNKNOWN:
                interpretation = replace(interpretation, status=InterpretationStatus.UNKNOWN, estimate=None, confidence=None, support=(), counter=())
            alternate_claim = replace(
                first_self_claim, kind=kind, meaning=meaning,
                interpretation=replace(interpretation, statement_id=meaning.statement_id(
                    domain="self_model", kind=kind.value
                )),
            )
            ordered = tuple(sorted((alternate_claim,) + self_model.claims[1:], key=lambda item: item.interpretation.statement_id))
            assert len(replace(self_model, claims=ordered).canonical_bytes()) <= max_self_model
