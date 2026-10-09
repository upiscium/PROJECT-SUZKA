"""Three independent, pure R15 U1 root schemas and their closed read shapes.

No constructor, JSON parser or checksum is an admission API. The later U2-U4
owners must validate current upstream roots and actual subject authorization
before staging any ordinary ordered event; U5 owns any durable v10 integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, cast

from suzka.limits import MAX_PERSISTED_REVISION
from suzka.r15.common import (
    MAX_CAPABILITIES,
    MAX_EPISODES,
    MAX_LINKS,
    MAX_NARRATIVE_CLAIMS,
    MAX_RELATIONSHIPS,
    MAX_REVISIONS,
    MAX_SELF_CLAIMS,
    SCHEMA_VERSION,
    Concept,
    Interpretation,
    InterpretationStatus,
    RevisionProof,
    SourceKind,
    SourceLifecycle,
    SourceDisposition,
    SourceWitness,
    canonical_json,
    check_root_digest,
    checksum,
    closed,
    digest,
    encode,
    exact_enum,
    identifier,
    integer,
    parse_enum,
    parse_json,
    root_value,
    sequence,
    source_disposition,
    validate_history,
)


RELATIONSHIP_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R15:RELATIONSHIP-ROOT:V1\0"
NARRATIVE_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R15:NARRATIVE-ROOT:V1\0"
SELF_MODEL_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R15:SELF-MODEL-ROOT:V1\0"


class SemanticReferentKind(str, Enum):
    """Which owner must resolve a proposition's precise bounded target."""

    EPISODE = "episode"
    EXPERIENCE = "experience"
    TASK_CLASS = "task_class"


class PropositionPolarity(str, Enum):
    AFFIRMS = "affirms"
    DENIES = "denies"


class SemanticCode(str, Enum):
    """Finite semantic atoms, not text, source proof or operator-authored traits."""

    NARRATIVE_CONTINUITY = "same_autobiographical_continuity"
    NARRATIVE_ROLE_ASSISTANT = "assistant_role_in_episode"
    NARRATIVE_ROLE_PARTNER = "conversation_partner_role_in_episode"
    NARRATIVE_REINTERPRETATION = "reinterpret_episode_as_subjectively_uncertain"
    NARRATIVE_CONTRADICTION = "conflicting_first_person_interpretations"
    SELF_IDENTITY_SUZKA = "subject_identity_suzka"
    SELF_ROLE_ASSISTANT = "subject_role_assistant"
    SELF_ROLE_PARTNER = "subject_role_conversation_partner"
    SELF_TRAIT_CAUTIOUS = "subject_possibly_cautious"
    SELF_TRAIT_PATIENT = "subject_possibly_patient"
    SELF_LIMITATION_UNVERIFIED_COMPETENCE = "subject_lacks_verified_task_competence"
    SELF_LIMITATION_DIRECT_ACTION = "subject_lacks_independent_external_action"
    SELF_EPISTEMIC_TASK_UNVERIFIED = "subject_task_competence_is_unknown"


@dataclass(frozen=True, slots=True)
class ClaimMeaning:
    """Closed proposition; scoped refs still require a trusted owner to resolve."""

    code: SemanticCode
    referent_kind: SemanticReferentKind
    referent: str
    polarity: PropositionPolarity

    def __post_init__(self) -> None:
        exact_enum(self.code, SemanticCode, "semantic code")
        exact_enum(self.referent_kind, SemanticReferentKind, "semantic referent kind")
        exact_enum(self.polarity, PropositionPolarity, "proposition polarity")
        if self.referent_kind is SemanticReferentKind.EPISODE:
            digest(self.referent, "episode referent")
        else:
            identifier(self.referent, "semantic referent")
        if self.code in (
            SemanticCode.SELF_LIMITATION_UNVERIFIED_COMPETENCE,
            SemanticCode.SELF_EPISTEMIC_TASK_UNVERIFIED,
            SemanticCode.SELF_LIMITATION_DIRECT_ACTION,
        ) and self.polarity is PropositionPolarity.DENIES:
            raise ValueError("denying this limitation would assert unverified capability")

    def statement_id(self, *, domain: str, kind: str, episode_ids: tuple[str, ...] = ()) -> str:
        self.__post_init__()
        if type(domain) is not str or domain not in ("narrative", "self_model"):
            raise ValueError("claim identity must use its closed owner domain")
        if type(kind) is not str or kind not in (
            *(member.value for member in NarrativeClaimKind),
            *(member.value for member in SelfClaimKind),
        ):
            raise ValueError("claim identity requires a closed claim kind")
        if type(episode_ids) is not tuple or len(episode_ids) > MAX_LINKS:
            raise ValueError("claim identity has invalid linked episodes")
        for episode_id in episode_ids:
            digest(episode_id, "claim linked episode")
        if domain == "self_model" and episode_ids:
            raise ValueError("SelfModel does not own narrative episode links")
        return checksum(
            b"PROJECT-SUZKA:R15:SEMANTIC-CLAIM:V1\0",
            {"domain": domain, "kind": kind, "meaning": encode(self), "episode_ids": list(episode_ids)},
        )

    @classmethod
    def from_value(cls, value: object) -> ClaimMeaning:
        row = closed(value, ("code", "referent_kind", "referent", "polarity"), "ClaimMeaning")
        if type(row["referent"]) is not str:
            raise TypeError("semantic referent must be an exact bounded string")
        return cls(
            parse_enum(row["code"], SemanticCode, "semantic code"),
            parse_enum(row["referent_kind"], SemanticReferentKind, "referent kind"),
            cast(str, row["referent"]),
            parse_enum(row["polarity"], PropositionPolarity, "polarity"),
        )


def current_semantic_producer(meaning: ClaimMeaning) -> SourceDisposition:
    """R09–R14 do not own these R15 propositions, even if syntax is valid.

    U3/U4 must establish and review an exact first-person mapping and ordered
    admission separately; R12's Experience is event lineage, not trait/role
    semantics. Never infer this proof from source text, a digest, or operator.
    """

    if type(meaning) is not ClaimMeaning:
        raise TypeError("semantic proposition must be exact")
    meaning.__post_init__()
    return SourceDisposition.UNAVAILABLE


class RelationshipAxisKind(str, Enum):
    TRUST = "trust"
    FAMILIARITY = "familiarity"
    CLOSENESS = "closeness"
    CAUTION = "caution"
    RECIPROCITY_EXPECTATION = "reciprocity_expectation"
    UNRESOLVED_ISSUE = "unresolved_issue"


@dataclass(frozen=True, slots=True)
class RelationshipAxis:
    kind: RelationshipAxisKind
    interpretation: Interpretation

    def __post_init__(self) -> None:
        exact_enum(self.kind, RelationshipAxisKind, "relationship axis")
        if type(self.interpretation) is not Interpretation:
            raise TypeError("axis interpretation must be exact")
        self.interpretation.__post_init__()
        self.interpretation.require_sources(Concept.RELATIONSHIP)

    @classmethod
    def from_value(cls, value: object) -> RelationshipAxis:
        row = closed(value, ("kind", "interpretation"), "RelationshipAxis")
        return cls(
            parse_enum(row["kind"], RelationshipAxisKind, "relationship axis"),
            Interpretation.from_value(row["interpretation"]),
        )


@dataclass(frozen=True, slots=True)
class RelationshipRecord:
    relationship_id: str
    interlocutor_key: str
    claimed_identity_key: str | None
    revision: int
    axes: tuple[RelationshipAxis, ...]

    def __post_init__(self) -> None:
        digest(self.relationship_id, "relationship_id")
        identifier(self.interlocutor_key, "interlocutor_key")
        if self.claimed_identity_key is not None:
            # This is the R09 *claim* and never verified personhood or merge.
            identifier(self.claimed_identity_key, "claimed_identity_key")
        integer(self.revision, "relationship revision", maximum=MAX_PERSISTED_REVISION)
        if type(self.axes) is not tuple or len(self.axes) != len(RelationshipAxisKind):
            raise ValueError("all independent relationship axes must be explicit")
        kinds: list[str] = []
        statements: set[str] = set()
        for axis in self.axes:
            if type(axis) is not RelationshipAxis:
                raise TypeError("relationship axis must be exact")
            axis.__post_init__()
            kinds.append(axis.kind.value)
            if axis.interpretation.statement_id in statements:
                raise ValueError("different axes cannot share an interpretation identity")
            statements.add(axis.interpretation.statement_id)
        if kinds != sorted(kind.value for kind in RelationshipAxisKind):
            raise ValueError("relationship axes must be complete and canonically ordered")

    @classmethod
    def from_value(cls, value: object) -> RelationshipRecord:
        row = closed(value, (
            "relationship_id", "interlocutor_key", "claimed_identity_key", "revision", "axes"
        ), "RelationshipRecord")
        return cls(
            digest(row["relationship_id"], "relationship_id"),
            identifier(row["interlocutor_key"], "interlocutor_key"),
            None if row["claimed_identity_key"] is None else identifier(row["claimed_identity_key"], "claimed_identity_key"),
            integer(row["revision"], "revision", maximum=MAX_PERSISTED_REVISION),
            tuple(RelationshipAxis.from_value(item) for item in sequence(row["axes"], "axes", len(RelationshipAxisKind), json_input=True)),
        )


@dataclass(frozen=True, slots=True)
class RelationshipState:
    schema_version: int
    revision: int
    records: tuple[RelationshipRecord, ...]
    history_anchor: str | None
    revision_history: tuple[RevisionProof, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported Relationship schema")
        if type(self.records) is not tuple or len(self.records) > MAX_RELATIONSHIPS:
            raise ValueError("Relationship records exceed their full legal limit")
        ids: list[str] = []
        interlocutors: set[str] = set()
        for record in self.records:
            if type(record) is not RelationshipRecord:
                raise TypeError("Relationship record must be exact")
            record.__post_init__()
            ids.append(record.relationship_id)
            if record.interlocutor_key in interlocutors:
                raise ValueError("interlocutor already has a separate relationship record")
            interlocutors.add(record.interlocutor_key)
            if record.revision > self.revision:
                raise ValueError("relationship record revision is in the future")
        if ids != sorted(set(ids)):
            raise ValueError("Relationship records must be sorted and unique")
        validate_history(self.revision, self.history_anchor, self.revision_history)

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return root_value(self, RELATIONSHIP_DOMAIN)

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_value(cls, value: object) -> RelationshipState:
        row = closed(value, (
            "schema_version", "revision", "records", "history_anchor", "revision_history", "state_digest"
        ), "RelationshipState")
        check_root_digest(row, RELATIONSHIP_DOMAIN)
        result = cls(
            integer(row["schema_version"], "schema_version", maximum=SCHEMA_VERSION, minimum=SCHEMA_VERSION),
            integer(row["revision"], "revision", maximum=MAX_PERSISTED_REVISION),
            tuple(RelationshipRecord.from_value(item) for item in sequence(row["records"], "records", MAX_RELATIONSHIPS, json_input=True)),
            None if row["history_anchor"] is None else digest(row["history_anchor"], "history_anchor"),
            tuple(RevisionProof.from_value(item) for item in sequence(row["revision_history"], "revision_history", MAX_REVISIONS, json_input=True)),
        )
        if result.canonical_value() != row:
            raise ValueError("Relationship value is not canonical")
        return result

    @classmethod
    def from_json(cls, payload: bytes) -> RelationshipState:
        from suzka.r15.bounds import derive_r15_schema_budget

        return cls.from_value(parse_json(payload, derive_r15_schema_budget().relationship_bytes))


class NarrativeLinkKind(str, Enum):
    EPISODE = "episode"
    RELATIONSHIP = "relationship"
    SELF_CLAIM = "self_claim"


@dataclass(frozen=True, slots=True)
class NarrativeLink:
    kind: NarrativeLinkKind
    target_id: str

    def __post_init__(self) -> None:
        exact_enum(self.kind, NarrativeLinkKind, "link kind")
        digest(self.target_id, "link target")

    @classmethod
    def from_value(cls, value: object) -> NarrativeLink:
        row = closed(value, ("kind", "target_id"), "NarrativeLink")
        return cls(parse_enum(row["kind"], NarrativeLinkKind, "kind"), digest(row["target_id"], "target_id"))


@dataclass(frozen=True, slots=True)
class NarrativeEpisode:
    episode_id: str
    experience: SourceWitness
    interpretation: Interpretation
    links: tuple[NarrativeLink, ...]

    def __post_init__(self) -> None:
        digest(self.episode_id, "episode_id")
        if type(self.experience) is not SourceWitness or self.experience.kind is not SourceKind.EXPERIENCE:
            raise ValueError("a narrative episode must refer to an R12 Experience")
        self.experience.__post_init__()
        if type(self.interpretation) is not Interpretation:
            raise TypeError("episode interpretation must be exact")
        self.interpretation.__post_init__()
        self.interpretation.require_sources(Concept.NARRATIVE_EPISODE)
        if self.experience.lifecycle is not SourceLifecycle.ACTIVE and self.interpretation.status is not InterpretationStatus.UNKNOWN:
            raise ValueError("an ineligible Experience cannot back a current episode")
        if self.interpretation.status is not InterpretationStatus.UNKNOWN and (
            self.experience not in self.interpretation.support
            or source_disposition(Concept.NARRATIVE_EPISODE, self.experience)
            is SourceDisposition.UNAVAILABLE
        ):
            raise ValueError("current episode requires its own eligible source in support")
        if type(self.links) is not tuple or len(self.links) > MAX_LINKS:
            raise ValueError("episode links exceed their bound")
        for link in self.links:
            if type(link) is not NarrativeLink:
                raise TypeError("episode link must be exact")
            link.__post_init__()
        keys = tuple((link.kind.value, link.target_id) for link in self.links)
        if keys != tuple(sorted(set(keys))) or any(
            link.kind is NarrativeLinkKind.EPISODE and link.target_id == self.episode_id for link in self.links
        ):
            raise ValueError("episode links must be ordered, unique, non-self-referential")

    @classmethod
    def from_value(cls, value: object) -> NarrativeEpisode:
        row = closed(value, ("episode_id", "experience", "interpretation", "links"), "NarrativeEpisode")
        return cls(
            digest(row["episode_id"], "episode_id"),
            SourceWitness.from_value(row["experience"]),
            Interpretation.from_value(row["interpretation"]),
            tuple(NarrativeLink.from_value(item) for item in sequence(row["links"], "links", MAX_LINKS, json_input=True)),
        )


class NarrativeClaimKind(str, Enum):
    CONTINUITY = "continuity"
    ROLE = "role"
    REINTERPRETATION = "reinterpretation"
    CONTRADICTION = "contradiction"


_NARRATIVE_CODES: Final[dict[NarrativeClaimKind, frozenset[SemanticCode]]] = {
    NarrativeClaimKind.CONTINUITY: frozenset({SemanticCode.NARRATIVE_CONTINUITY}),
    NarrativeClaimKind.ROLE: frozenset({
        SemanticCode.NARRATIVE_ROLE_ASSISTANT, SemanticCode.NARRATIVE_ROLE_PARTNER,
    }),
    NarrativeClaimKind.REINTERPRETATION: frozenset({SemanticCode.NARRATIVE_REINTERPRETATION}),
    NarrativeClaimKind.CONTRADICTION: frozenset({SemanticCode.NARRATIVE_CONTRADICTION}),
}


@dataclass(frozen=True, slots=True)
class NarrativeClaim:
    kind: NarrativeClaimKind
    interpretation: Interpretation
    episode_ids: tuple[str, ...]
    meaning: ClaimMeaning

    def __post_init__(self) -> None:
        exact_enum(self.kind, NarrativeClaimKind, "narrative kind")
        if type(self.interpretation) is not Interpretation:
            raise TypeError("narrative claim interpretation must be exact")
        self.interpretation.__post_init__()
        self.interpretation.require_sources(Concept.NARRATIVE_CLAIM)
        if type(self.meaning) is not ClaimMeaning:
            raise TypeError("narrative meaning must be exact")
        self.meaning.__post_init__()
        if (
            self.meaning.code not in _NARRATIVE_CODES[self.kind]
            or self.meaning.referent_kind is not SemanticReferentKind.EPISODE
        ):
            raise ValueError("narrative kind and source-owned episode meaning disagree")
        if type(self.episode_ids) is not tuple or len(self.episode_ids) > MAX_LINKS:
            raise ValueError("narrative episode references exceed their bound")
        for item in self.episode_ids:
            digest(item, "episode ref")
        if self.episode_ids != tuple(sorted(set(self.episode_ids))):
            raise ValueError("narrative episode references must be ordered and unique")
        if self.meaning.referent not in self.episode_ids:
            raise ValueError("narrative proposition's episode referent is missing")
        if self.kind is NarrativeClaimKind.CONTINUITY and len(self.episode_ids) < 2:
            raise ValueError("autobiographical continuity must name two distinct episodes")
        if self.interpretation.statement_id != self.meaning.statement_id(
            domain="narrative", kind=self.kind.value, episode_ids=self.episode_ids
        ):
            raise ValueError("narrative statement identity does not bind its meaning and links")

    @classmethod
    def from_value(cls, value: object) -> NarrativeClaim:
        row = closed(value, ("kind", "interpretation", "episode_ids", "meaning"), "NarrativeClaim")
        return cls(
            parse_enum(row["kind"], NarrativeClaimKind, "kind"),
            Interpretation.from_value(row["interpretation"]),
            tuple(digest(item, "episode ref") for item in sequence(row["episode_ids"], "episode_ids", MAX_LINKS, json_input=True)),
            ClaimMeaning.from_value(row["meaning"]),
        )


@dataclass(frozen=True, slots=True)
class NarrativeSelfState:
    schema_version: int
    revision: int
    episodes: tuple[NarrativeEpisode, ...]
    claims: tuple[NarrativeClaim, ...]
    history_anchor: str | None
    revision_history: tuple[RevisionProof, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported NarrativeSelf schema")
        if type(self.episodes) is not tuple or len(self.episodes) > MAX_EPISODES or type(self.claims) is not tuple or len(self.claims) > MAX_NARRATIVE_CLAIMS:
            raise ValueError("NarrativeSelf content exceeds its full legal limit")
        for item in self.episodes:
            if type(item) is not NarrativeEpisode:
                raise TypeError("narrative episode must be exact")
            item.__post_init__()
        for claim_item in self.claims:
            if type(claim_item) is not NarrativeClaim:
                raise TypeError("narrative claim must be exact")
            claim_item.__post_init__()
        ids = tuple(item.episode_id for item in self.episodes)
        claim_ids = tuple(item.interpretation.statement_id for item in self.claims)
        if ids != tuple(sorted(set(ids))) or claim_ids != tuple(sorted(set(claim_ids))):
            raise ValueError("narrative episodes and claims must be sorted and unique")
        if any(not set(item.episode_ids).issubset(ids) for item in self.claims):
            raise ValueError("narrative claim links must target retained episodes")
        episodes_by_id = {item.episode_id: item for item in self.episodes}
        for claim_item in self.claims:
            if claim_item.interpretation.status is not InterpretationStatus.UNKNOWN:
                if any(
                    episodes_by_id[episode_id].experience.lifecycle is not SourceLifecycle.ACTIVE
                    or episodes_by_id[episode_id].interpretation.status is InterpretationStatus.UNKNOWN
                    for episode_id in claim_item.episode_ids
                ):
                    raise ValueError("a current narrative proposition cannot use an unavailable episode")
        validate_history(self.revision, self.history_anchor, self.revision_history)

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return root_value(self, NARRATIVE_DOMAIN)

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_value(cls, value: object) -> NarrativeSelfState:
        row = closed(value, (
            "schema_version", "revision", "episodes", "claims", "history_anchor", "revision_history", "state_digest"
        ), "NarrativeSelfState")
        check_root_digest(row, NARRATIVE_DOMAIN)
        result = cls(
            integer(row["schema_version"], "schema_version", maximum=SCHEMA_VERSION, minimum=SCHEMA_VERSION),
            integer(row["revision"], "revision", maximum=MAX_PERSISTED_REVISION),
            tuple(NarrativeEpisode.from_value(item) for item in sequence(row["episodes"], "episodes", MAX_EPISODES, json_input=True)),
            tuple(NarrativeClaim.from_value(item) for item in sequence(row["claims"], "claims", MAX_NARRATIVE_CLAIMS, json_input=True)),
            None if row["history_anchor"] is None else digest(row["history_anchor"], "history_anchor"),
            tuple(RevisionProof.from_value(item) for item in sequence(row["revision_history"], "revision_history", MAX_REVISIONS, json_input=True)),
        )
        if result.canonical_value() != row:
            raise ValueError("NarrativeSelf value is not canonical")
        return result

    @classmethod
    def from_json(cls, payload: bytes) -> NarrativeSelfState:
        from suzka.r15.bounds import derive_r15_schema_budget

        return cls.from_value(parse_json(payload, derive_r15_schema_budget().narrative_bytes))


class SelfClaimKind(str, Enum):
    IDENTITY = "identity"
    ROLE = "role"
    POSSIBLE_TRAIT = "possible_trait"
    LIMITATION = "limitation"
    EPISTEMIC_UNKNOWN = "epistemic_unknown"


_SELF_CODES: Final[dict[SelfClaimKind, frozenset[SemanticCode]]] = {
    SelfClaimKind.IDENTITY: frozenset({SemanticCode.SELF_IDENTITY_SUZKA}),
    SelfClaimKind.ROLE: frozenset({SemanticCode.SELF_ROLE_ASSISTANT, SemanticCode.SELF_ROLE_PARTNER}),
    SelfClaimKind.POSSIBLE_TRAIT: frozenset({
        SemanticCode.SELF_TRAIT_CAUTIOUS, SemanticCode.SELF_TRAIT_PATIENT,
    }),
    SelfClaimKind.LIMITATION: frozenset({
        SemanticCode.SELF_LIMITATION_UNVERIFIED_COMPETENCE,
        SemanticCode.SELF_LIMITATION_DIRECT_ACTION,
    }),
    SelfClaimKind.EPISTEMIC_UNKNOWN: frozenset({SemanticCode.SELF_EPISTEMIC_TASK_UNVERIFIED}),
}


@dataclass(frozen=True, slots=True)
class SelfClaim:
    kind: SelfClaimKind
    interpretation: Interpretation
    meaning: ClaimMeaning

    def __post_init__(self) -> None:
        exact_enum(self.kind, SelfClaimKind, "self claim kind")
        if type(self.interpretation) is not Interpretation:
            raise TypeError("self interpretation must be exact")
        self.interpretation.__post_init__()
        self.interpretation.require_sources(Concept.SELF_HYPOTHESIS)
        if type(self.meaning) is not ClaimMeaning:
            raise TypeError("self meaning must be exact")
        self.meaning.__post_init__()
        if self.meaning.code not in _SELF_CODES[self.kind]:
            raise ValueError("self-claim kind and exact semantic predicate disagree")
        required_kind = (
            SemanticReferentKind.TASK_CLASS
            if self.kind is SelfClaimKind.EPISTEMIC_UNKNOWN
            else SemanticReferentKind.EXPERIENCE
        )
        if self.meaning.referent_kind is not required_kind:
            raise ValueError("self claim's semantic referent has an invalid owner")
        if self.meaning.referent_kind is SemanticReferentKind.TASK_CLASS:
            if self.interpretation.status is not InterpretationStatus.UNKNOWN:
                raise ValueError("unverified task competence cannot be a present self fact")
        elif self.interpretation.status is not InterpretationStatus.UNKNOWN and not any(
            item.kind is SourceKind.EXPERIENCE and item.reference == self.meaning.referent
            for item in self.interpretation.support
        ):
            raise ValueError("a current self proposition lacks its source-owned referent")
        if self.interpretation.statement_id != self.meaning.statement_id(
            domain="self_model", kind=self.kind.value
        ):
            raise ValueError("self statement identity does not bind its exact meaning")

    @classmethod
    def from_value(cls, value: object) -> SelfClaim:
        row = closed(value, ("kind", "interpretation", "meaning"), "SelfClaim")
        return cls(
            parse_enum(row["kind"], SelfClaimKind, "self claim kind"),
            Interpretation.from_value(row["interpretation"]),
            ClaimMeaning.from_value(row["meaning"]),
        )


class VerifiedCompetence(str, Enum):
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CapabilityHypothesis:
    task_class_key: str
    attempted: Interpretation
    verified_competence: VerifiedCompetence

    def __post_init__(self) -> None:
        identifier(self.task_class_key, "task_class_key")
        if type(self.attempted) is not Interpretation:
            raise TypeError("task-attempt hypothesis must be exact")
        self.attempted.__post_init__()
        self.attempted.require_sources(Concept.TASK_ATTEMPT)
        if type(self.verified_competence) is not VerifiedCompetence or self.verified_competence is not VerifiedCompetence.UNKNOWN:
            raise ValueError("no qualified verified-competence producer exists before R17/R18")

    @classmethod
    def from_value(cls, value: object) -> CapabilityHypothesis:
        row = closed(value, ("task_class_key", "attempted", "verified_competence"), "CapabilityHypothesis")
        return cls(
            identifier(row["task_class_key"], "task_class_key"),
            Interpretation.from_value(row["attempted"]),
            parse_enum(row["verified_competence"], VerifiedCompetence, "verified_competence"),
        )


@dataclass(frozen=True, slots=True)
class SelfModelState:
    schema_version: int
    revision: int
    claims: tuple[SelfClaim, ...]
    capabilities: tuple[CapabilityHypothesis, ...]
    history_anchor: str | None
    revision_history: tuple[RevisionProof, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported SelfModel schema")
        if type(self.claims) is not tuple or len(self.claims) > MAX_SELF_CLAIMS or type(self.capabilities) is not tuple or len(self.capabilities) > MAX_CAPABILITIES:
            raise ValueError("SelfModel content exceeds its full legal limit")
        for item in self.claims:
            if type(item) is not SelfClaim:
                raise TypeError("self claim must be exact")
            item.__post_init__()
        for capability_item in self.capabilities:
            if type(capability_item) is not CapabilityHypothesis:
                raise TypeError("capability must be exact")
            capability_item.__post_init__()
        ids = tuple(item.interpretation.statement_id for item in self.claims)
        tasks = tuple(item.task_class_key for item in self.capabilities)
        if ids != tuple(sorted(set(ids))) or tasks != tuple(sorted(set(tasks))):
            raise ValueError("self claims and task classes must be sorted and unique")
        if any(
            item.meaning.referent not in tasks
            for item in self.claims
            if item.meaning.referent_kind is SemanticReferentKind.TASK_CLASS
        ):
            raise ValueError("SelfModel proposition targets a missing task class")
        validate_history(self.revision, self.history_anchor, self.revision_history)

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return root_value(self, SELF_MODEL_DOMAIN)

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_value(cls, value: object) -> SelfModelState:
        row = closed(value, (
            "schema_version", "revision", "claims", "capabilities", "history_anchor", "revision_history", "state_digest"
        ), "SelfModelState")
        check_root_digest(row, SELF_MODEL_DOMAIN)
        result = cls(
            integer(row["schema_version"], "schema_version", maximum=SCHEMA_VERSION, minimum=SCHEMA_VERSION),
            integer(row["revision"], "revision", maximum=MAX_PERSISTED_REVISION),
            tuple(SelfClaim.from_value(item) for item in sequence(row["claims"], "claims", MAX_SELF_CLAIMS, json_input=True)),
            tuple(CapabilityHypothesis.from_value(item) for item in sequence(row["capabilities"], "capabilities", MAX_CAPABILITIES, json_input=True)),
            None if row["history_anchor"] is None else digest(row["history_anchor"], "history_anchor"),
            tuple(RevisionProof.from_value(item) for item in sequence(row["revision_history"], "revision_history", MAX_REVISIONS, json_input=True)),
        )
        if result.canonical_value() != row:
            raise ValueError("SelfModel value is not canonical")
        return result

    @classmethod
    def from_json(cls, payload: bytes) -> SelfModelState:
        from suzka.r15.bounds import derive_r15_schema_budget

        return cls.from_value(parse_json(payload, derive_r15_schema_budget().self_model_bytes))
