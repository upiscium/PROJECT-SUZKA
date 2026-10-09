"""Read-only, full-legal R15 U1 schema capacity derivation (NOT a v10 codec).

The fixtures instantiate every legal row slot, all retained evidence and
contrary windows, all links and revision proofs. All strings are bounded ASCII
identifiers, fixed hex digests, closed enums or fixed UTC timestamps; numbers
are bounded integers. Choosing maximal-width legal values yields an envelope
for each field; IDs are varied without shortening them to satisfy uniqueness.
This is a schema-size proof, not validation of actual source authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from typing import Final

from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
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
    SCORE_SCALE,
    Accessibility,
    ClaimedOrigin,
    EventRef,
    Inertia,
    Interpretation,
    InterpretationStatus,
    RevisionProof,
    RevisionReason,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
    canonical_json,
)
from suzka.r15.contracts import (
    CapabilityHypothesis,
    ClaimMeaning,
    NarrativeClaim,
    NarrativeClaimKind,
    NarrativeEpisode,
    NarrativeLink,
    NarrativeLinkKind,
    NarrativeSelfState,
    RelationshipAxis,
    RelationshipAxisKind,
    RelationshipRecord,
    RelationshipState,
    SelfClaim,
    SelfClaimKind,
    SelfModelState,
    SemanticCode,
    SemanticReferentKind,
    PropositionPolarity,
    VerifiedCompetence,
)


# Frozen exact integrated R14 v9 accounting. Tests independently compare the
# actual runtime v9 projector and the v9 root field set; the pure module never
# imports runtime, to avoid acquiring a state-port or WAL dependency.
V9_BASE_MAX_BYTES: Final = 113_767_203
FULL_FUTURE_RESERVE_BYTES: Final = 16_777_216
V9_WITH_RESERVE_BYTES: Final = V9_BASE_MAX_BYTES + FULL_FUTURE_RESERVE_BYTES
AGENT_STATE_HARD_CAP_BYTES: Final = 134_217_728
R15_ROOT_NAMES: Final = (
    "relationship_state", "narrative_self_state", "self_model_state"
)
_TIME: Final = datetime.max.replace(tzinfo=UTC)
_DIGEST: Final = "f" * 64


def _hex_id(index: int) -> str:
    return f"{index:064x}"


def _identifier(prefix: str, index: int) -> str:
    return prefix * 120 + f"{index:08x}"


def _event(index: int) -> EventRef:
    return EventRef(_identifier("e", index), MAX_PERSISTED_EVENT_SEQUENCE - MAX_REVISIONS + index + 1, _TIME)


def _source(index: int) -> SourceWitness:
    # A full-width reference, exact event, max-width counters/digests and the
    # longest statically permissible first-person evidence source for all
    # three current-owner hypotheses; this still requires trusted R12 commit.
    return SourceWitness(
        SourceKind.EXPERIENCE,
        _identifier("x", index),
        MAX_PERSISTED_REVISION,
        _DIGEST,
        _DIGEST,
        SourceLifecycle.ACTIVE,
        ClaimedOrigin.SUBJECT_OBSERVATION,
        _event(MAX_REVISIONS - 1),
    )


def _interpretation(index: int) -> Interpretation:
    return Interpretation(
        _hex_id(index),
        InterpretationStatus.CONTESTED,
        SCORE_SCALE,
        SCORE_SCALE,
        tuple(_source(index * (MAX_SUPPORT + MAX_COUNTER) + offset) for offset in range(MAX_SUPPORT)),
        tuple(_source(index * (MAX_SUPPORT + MAX_COUNTER) + MAX_SUPPORT + offset) for offset in range(MAX_COUNTER)),
        Accessibility.REACTIVATABLE,
        Inertia.RESISTANT,
    )


def _history(target: str) -> tuple[RevisionProof, ...]:
    result: list[RevisionProof] = []
    previous = _DIGEST
    reason = max(RevisionReason, key=lambda item: len(item.value))
    for position in range(MAX_REVISIONS):
        proof = RevisionProof(
            MAX_PERSISTED_REVISION - MAX_REVISIONS + position + 1,
            _event(position),
            target,
            _DIGEST,
            _DIGEST,
            tuple(_hex_id(index) for index in range(MAX_SUPPORT)),
            tuple(_hex_id(index + MAX_SUPPORT) for index in range(MAX_COUNTER)),
            reason,
            previous,
        )
        result.append(proof)
        previous = proof.record_digest
    return tuple(result)


@lru_cache(maxsize=1)
def maximum_legal_r15_roots() -> tuple[RelationshipState, NarrativeSelfState, SelfModelState]:
    """Materialize all three full-bound schemas, never a selected small sample."""

    relationship_records = tuple(
        RelationshipRecord(
            _hex_id(index),
            _identifier("r", index),
            _identifier("i", index),
            MAX_PERSISTED_REVISION,
            tuple(
                RelationshipAxis(kind, _interpretation(index * len(RelationshipAxisKind) + offset))
                for offset, kind in enumerate(sorted(RelationshipAxisKind, key=lambda kind: kind.value))
            ),
        )
        for index in range(MAX_RELATIONSHIPS)
    )
    relationship = RelationshipState(
        1, MAX_PERSISTED_REVISION, relationship_records, _DIGEST,
        _history(relationship_records[0].axes[0].interpretation.statement_id),
    )

    episode_ids = tuple(_hex_id(index + 10000) for index in range(MAX_EPISODES))
    episodes = tuple(
        NarrativeEpisode(
            episode_ids[index],
            _source(20000 + index),
            Interpretation(
                _hex_id(30000 + index),
                InterpretationStatus.CONTESTED,
                SCORE_SCALE,
                SCORE_SCALE,
                (_source(20000 + index),) + tuple(
                    _source((30000 + index) * (MAX_SUPPORT + MAX_COUNTER) + offset)
                    for offset in range(MAX_SUPPORT - 1)
                ),
                tuple(
                    _source((30000 + index) * (MAX_SUPPORT + MAX_COUNTER) + MAX_SUPPORT + offset)
                    for offset in range(MAX_COUNTER)
                ),
                Accessibility.REACTIVATABLE,
                Inertia.RESISTANT,
            ),
            tuple(
                NarrativeLink(NarrativeLinkKind.RELATIONSHIP, _hex_id(40000 + offset))
                for offset in range(MAX_LINKS)
            ),
        )
        for index in range(MAX_EPISODES)
    )
    claims = tuple(
        NarrativeClaim(
            NarrativeClaimKind.REINTERPRETATION,
            replace(
                _interpretation(50000 + index),
                statement_id=ClaimMeaning(
                    SemanticCode.NARRATIVE_REINTERPRETATION,
                    SemanticReferentKind.EPISODE,
                    episode_ids[index],
                    PropositionPolarity.AFFIRMS,
                ).statement_id(
                    domain="narrative",
                    kind=NarrativeClaimKind.REINTERPRETATION.value,
                    episode_ids=episode_ids[index:index + MAX_LINKS],
                ),
            ),
            episode_ids[index:index + MAX_LINKS],
            ClaimMeaning(
                SemanticCode.NARRATIVE_REINTERPRETATION,
                SemanticReferentKind.EPISODE,
                episode_ids[index],
                PropositionPolarity.AFFIRMS,
            ),
        )
        for index in range(MAX_NARRATIVE_CLAIMS)
    )
    ordered_claims = tuple(sorted(claims, key=lambda item: item.interpretation.statement_id))
    narrative = NarrativeSelfState(
        1, MAX_PERSISTED_REVISION, episodes, ordered_claims, _DIGEST,
        _history(ordered_claims[0].interpretation.statement_id),
    )

    self_claims = tuple(
        SelfClaim(
            SelfClaimKind.LIMITATION,
            replace(
                _interpretation(60000 + index),
                statement_id=ClaimMeaning(
                    SemanticCode.SELF_LIMITATION_DIRECT_ACTION,
                    SemanticReferentKind.EXPERIENCE,
                    _interpretation(60000 + index).support[0].reference,
                    PropositionPolarity.AFFIRMS,
                ).statement_id(
                    domain="self_model", kind=SelfClaimKind.LIMITATION.value
                ),
            ),
            ClaimMeaning(
                SemanticCode.SELF_LIMITATION_DIRECT_ACTION,
                SemanticReferentKind.EXPERIENCE,
                _interpretation(60000 + index).support[0].reference,
                PropositionPolarity.AFFIRMS,
            ),
        )
        for index in range(MAX_SELF_CLAIMS)
    )
    capabilities = tuple(
        CapabilityHypothesis(
            _identifier("t", index),
            _interpretation(70000 + index),
            VerifiedCompetence.UNKNOWN,
        )
        for index in range(MAX_CAPABILITIES)
    )
    ordered_self_claims = tuple(sorted(self_claims, key=lambda item: item.interpretation.statement_id))
    self_model = SelfModelState(
        1, MAX_PERSISTED_REVISION, ordered_self_claims, capabilities, _DIGEST,
        _history(ordered_self_claims[0].interpretation.statement_id),
    )
    return relationship, narrative, self_model


@dataclass(frozen=True, slots=True)
class R15SchemaBudget:
    relationship_bytes: int
    narrative_bytes: int
    self_model_bytes: int
    root_overhead_bytes: int
    v9_schema_version_delta_bytes: int
    v10_before_reserve_bytes: int
    v10_with_reserve_bytes: int
    remaining_beyond_reserve_bytes: int


@lru_cache(maxsize=1)
def derive_r15_schema_budget() -> R15SchemaBudget:
    relationship, narrative, self_model = maximum_legal_r15_roots()
    sizes = (
        len(relationship.canonical_bytes()),
        len(narrative.canonical_bytes()),
        len(self_model.canonical_bytes()),
    )
    overhead = sum(len(canonical_json(name)) + 2 for name in R15_ROOT_NAMES)
    version_delta = len(canonical_json(10)) - len(canonical_json(9))
    base = V9_BASE_MAX_BYTES + sum(sizes) + overhead + version_delta
    with_reserve = base + FULL_FUTURE_RESERVE_BYTES
    return R15SchemaBudget(
        *sizes, overhead, version_delta, base, with_reserve,
        AGENT_STATE_HARD_CAP_BYTES - with_reserve,
    )


def require_r15_capacity() -> R15SchemaBudget:
    result = derive_r15_schema_budget()
    if result.remaining_beyond_reserve_bytes < 0:
        raise ValueError(
            f"CAPACITY DESIGN HOLD: full legal R15 v10 with 16 MiB reserve "
            f"exceeds 128 MiB by {-result.remaining_beyond_reserve_bytes} bytes"
        )
    return result
