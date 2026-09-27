"""Request-scoped semantic provenance evidence.

This module is deliberately a small bridge between the DB2 projection and the
semantic authority.  It produces no durable state and never resolves legacy
source ids through DB1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from suzka.identifiers import validate_identifier
from suzka.memory.semantic_lifecycle import (
    SEMANTIC_MAX_SOURCE_EDGES,
    SemanticProvenanceClass,
    SemanticRevision,
    SemanticSourceEdge,
    SemanticSourceKind,
    SemanticSourceStatus,
    canonicalize_source_edges,
    provenance_for_edges,
)
from suzka.memory.semantic_store import (
    SemanticStore,
    SemanticStoreConflict,
    SemanticStoreCorrupt,
    SemanticStoreUnavailable,
)


_R12_PROJECTION_FIELDS = frozenset(
    {
        "semantic_projection_schema",
        "semantic_revision",
        "semantic_revision_digest",
        "semantic_content_digest",
        "semantic_lifecycle",
        "semantic_provenance_digest",
    }
)


class SemanticContextProjectionUnavailable(Exception):
    """The authoritative revision could not be read."""


class SemanticContextProjectionInvalid(Exception):
    """DB2 or authoritative semantic evidence is malformed or divergent."""


@dataclass(frozen=True, slots=True)
class SemanticContextEvidence:
    """Immutable, request-scoped evidence suitable for contextual projection."""

    document: str
    semantic_id: str
    provenance_class: SemanticProvenanceClass
    source_count: int
    source_edges: tuple[SemanticSourceEdge, ...]
    cross_context: bool = False
    unknown_or_incomplete: bool = False
    authoritative_revision: int | None = None
    authoritative_digest: str | None = None
    source_less: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_id", validate_identifier(self.semantic_id))
        if not isinstance(self.document, str):
            raise TypeError("document must be a string")
        if not isinstance(self.provenance_class, SemanticProvenanceClass):
            raise TypeError("provenance_class must be SemanticProvenanceClass")
        edges = canonicalize_source_edges(self.source_edges)
        if edges != self.source_edges:
            object.__setattr__(self, "source_edges", edges)
        if (
            isinstance(self.source_count, bool)
            or not isinstance(self.source_count, int)
            or self.source_count != len(edges)
        ):
            raise ValueError("source_count does not match source_edges")
        if provenance_for_edges(edges).classification is not self.provenance_class:
            raise ValueError("provenance_class does not match source_edges")
        if not isinstance(self.cross_context, bool) or not isinstance(
            self.unknown_or_incomplete, bool
        ):
            raise TypeError("projection flags must be bool")
        if self.cross_context != (
            self.provenance_class is SemanticProvenanceClass.MULTI_CONTEXT
        ):
            raise ValueError("cross_context does not match provenance_class")
        if self.unknown_or_incomplete != (
            self.provenance_class
            in (SemanticProvenanceClass.UNKNOWN, SemanticProvenanceClass.INCOMPLETE)
        ):
            raise ValueError("unknown_or_incomplete does not match provenance_class")
        if not isinstance(self.source_less, bool) or self.source_less != (not edges):
            raise ValueError("source_less does not match source_edges")
        if (self.authoritative_revision is None) != (
            self.authoritative_digest is None
        ):
            raise ValueError("authority revision and digest must be bound together")
        if self.authoritative_revision is not None and (
            isinstance(self.authoritative_revision, bool)
            or not isinstance(self.authoritative_revision, int)
            or self.authoritative_revision < 0
        ):
            raise ValueError("authoritative_revision must be non-negative")
        if self.authoritative_digest is not None:
            if (
                not isinstance(self.authoritative_digest, str)
                or len(self.authoritative_digest) != 64
                or any(character not in "0123456789abcdef" for character in self.authoritative_digest)
            ):
                raise ValueError("authoritative_digest must be a lowercase SHA-256 digest")

    @property
    def provenance_classification(self) -> SemanticProvenanceClass:
        return self.provenance_class

    @property
    def canonical_source_edges(self) -> tuple[SemanticSourceEdge, ...]:
        return self.source_edges

    @property
    def revision(self) -> int | None:
        return self.authoritative_revision

    @property
    def revision_digest(self) -> str | None:
        return self.authoritative_digest


def semantic_context_evidence(
    semantic_id: str,
    document: object,
    metadata: Mapping[str, Any],
    store: SemanticStore,
) -> SemanticContextEvidence:
    """Build evidence from one already-read DB2 row.

    New-format rows are checked against the exact retained authority revision.
    Legacy rows use only their retained DB2 metadata and intentionally do not
    consult DB1.
    """

    try:
        semantic_id = validate_identifier(semantic_id)
    except (TypeError, ValueError):
        raise SemanticContextProjectionInvalid("Semantic identity is invalid") from None
    if not isinstance(document, str) or not isinstance(metadata, Mapping):
        raise SemanticContextProjectionInvalid("Semantic projection is invalid")

    schema = metadata.get("semantic_projection_schema")
    if schema == "r12.semantic.v1":
        return _new_format_evidence(semantic_id, document, metadata, store)
    if has_r12_projection_fields(metadata):
        raise SemanticContextProjectionInvalid(
            "Semantic projection marker or fields are invalid"
        )
    return _legacy_evidence(semantic_id, document, metadata)


def has_r12_projection_fields(metadata: Mapping[str, Any]) -> bool:
    """Return whether metadata carries any R12-only projection marker fields."""

    return bool(_R12_PROJECTION_FIELDS.intersection(metadata))


def _new_format_evidence(
    semantic_id: str,
    document: str,
    metadata: Mapping[str, Any],
    store: SemanticStore,
) -> SemanticContextEvidence:
    revision = metadata.get("semantic_revision")
    if type(revision) is not int or revision < 0:
        raise SemanticContextProjectionInvalid("Semantic revision is invalid")
    try:
        retained = store.load_current(semantic_id)
    except SemanticStoreUnavailable as error:
        raise SemanticContextProjectionUnavailable(
            "Authoritative Semantic revision is unavailable"
        ) from error
    except (SemanticStoreConflict, SemanticStoreCorrupt, ValueError) as error:
        raise SemanticContextProjectionInvalid(
            "Authoritative Semantic revision is invalid"
        ) from error
    if retained is None or retained.revision.revision != revision:
        raise SemanticContextProjectionInvalid(
            "Authoritative Semantic revision is not current"
        )
    authority = retained.revision
    expected = {
        "semantic_projection_schema": "r12.semantic.v1",
        "semantic_revision": authority.revision,
        "semantic_revision_digest": authority.revision_digest,
        "semantic_content_digest": authority.content_digest,
        "semantic_lifecycle": authority.lifecycle.value,
        "semantic_provenance_digest": authority.provenance.digest,
        "text": authority.semantic_content,
        "source_episode_ids": metadata.get("source_episode_ids"),
        "record_type": metadata.get("record_type"),
        "created_at": metadata.get("created_at"),
        "extra": metadata.get("extra"),
    }
    # Compare the complete projection metadata, including the canonical source
    # summary, without treating that summary as provenance authority.
    actual = dict(metadata)
    expected.update(_projection_metadata_tail(authority))
    if (
        semantic_id != authority.semantic_id
        or document != authority.semantic_content
        or actual != expected
    ):
        raise SemanticContextProjectionInvalid("Semantic DB2 projection is not authoritative")
    edges = authority.source_edges
    return SemanticContextEvidence(
        document=authority.semantic_content,
        semantic_id=semantic_id,
        provenance_class=authority.provenance.classification,
        source_count=authority.provenance.source_count,
        source_edges=edges,
        cross_context=(
            authority.provenance.classification is SemanticProvenanceClass.MULTI_CONTEXT
        ),
        unknown_or_incomplete=authority.provenance.classification
        in (SemanticProvenanceClass.UNKNOWN, SemanticProvenanceClass.INCOMPLETE),
        authoritative_revision=authority.revision,
        authoritative_digest=authority.revision_digest,
        source_less=not edges,
    )


def _projection_metadata_tail(revision: SemanticRevision) -> dict[str, object]:
    import json
    from datetime import UTC
    from suzka.memory.memory_schema import MemoryRecordType

    ids = sorted(
        {
            edge.source_id
            for edge in revision.source_edges
            if edge.source_kind is SemanticSourceKind.EPISODIC
        }
    )
    result: dict[str, object] = {
        "source_episode_ids": json.dumps(ids, ensure_ascii=True, separators=(",", ":")),
        "record_type": MemoryRecordType.SEMANTIC_MEMORY.value,
        "created_at": revision.created_at.astimezone(UTC).isoformat(timespec="microseconds"),
        "extra": "{}",
    }
    if revision.provenance.classification is SemanticProvenanceClass.SINGLE_CONTEXT:
        result["context_id"] = revision.provenance.known_context_ids[0]
    return result


def _legacy_evidence(
    semantic_id: str, document: str, metadata: Mapping[str, Any]
) -> SemanticContextEvidence:
    import json

    raw_sources = metadata.get("source_episode_ids")
    if not isinstance(raw_sources, str):
        raise SemanticContextProjectionInvalid("Legacy source evidence is invalid")
    try:
        sources = json.loads(raw_sources)
    except (TypeError, json.JSONDecodeError):
        raise SemanticContextProjectionInvalid("Legacy source evidence is invalid") from None
    if not isinstance(sources, list) or any(type(item) is not str for item in sources):
        raise SemanticContextProjectionInvalid("Legacy source evidence is invalid")
    if len(sources) > SEMANTIC_MAX_SOURCE_EDGES:
        raise SemanticContextProjectionInvalid("Legacy source evidence exceeds its bound")
    try:
        unique_sources = tuple(sorted({validate_identifier(item) for item in sources}))
        context = metadata.get("context_id")
        if context is not None:
            context = validate_identifier(context)
    except (TypeError, ValueError):
        raise SemanticContextProjectionInvalid("Legacy source evidence is invalid") from None
    if context is not None and not unique_sources:
        raise SemanticContextProjectionInvalid("Legacy context has no source evidence")
    edges = tuple(
        SemanticSourceEdge(
            source_kind=SemanticSourceKind.EPISODIC,
            source_id=source_id,
            captured_context_id=context,
            source_status=SemanticSourceStatus.AVAILABLE,
        )
        for source_id in unique_sources
    )
    classification = (
        SemanticProvenanceClass.UNKNOWN
        if not context
        else SemanticProvenanceClass.SINGLE_CONTEXT
    )
    return SemanticContextEvidence(
        document=document,
        semantic_id=semantic_id,
        provenance_class=classification,
        source_count=len(edges),
        source_edges=edges,
        unknown_or_incomplete=not context,
        source_less=not edges,
    )


__all__ = [
    "SemanticContextEvidence",
    "SemanticContextProjectionInvalid",
    "SemanticContextProjectionUnavailable",
    "has_r12_projection_fields",
    "SemanticEvidence",
    "build_semantic_context_evidence",
    "semantic_context_evidence",
]

SemanticEvidence = SemanticContextEvidence
build_semantic_context_evidence = semantic_context_evidence
