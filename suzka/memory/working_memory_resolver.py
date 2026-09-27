"""Bridge committed dual-memory reads into the runtime resolution contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from suzka.identifiers import validate_identifier
from suzka.memory.memory_schema import SemanticMemoryRecord
from suzka.memory.semantic_lifecycle import (
    SemanticProvenanceClass,
    SemanticRevision,
    SemanticSourceKind,
    SemanticSourceStatus,
)
from suzka.runtime.context import ContextRegistry, ContextRelation
from suzka.runtime.working_memory import (
    WorkingMemoryItem,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemorySourceKind,
)

if TYPE_CHECKING:
    from suzka.memory.dual_memory_system import DualMemorySystem


@dataclass(frozen=True, slots=True)
class SemanticContextSourceEvidence:
    """Request-scoped Context relation evidence for one canonical Semantic source."""

    source_kind: SemanticSourceKind
    source_id: str
    source_revision: int | None
    source_status: SemanticSourceStatus
    captured_context_id: str | None
    relation: ContextRelation
    compatibility_score: float


@dataclass(frozen=True, slots=True)
class SemanticContextProjection:
    """Pure multi-Context compatibility projection for one Semantic record."""

    semantic_id: str
    provenance_classification: SemanticProvenanceClass
    source_count: int
    source_evidence: tuple[SemanticContextSourceEvidence, ...]
    aggregate_compatibility: float
    cross_context: bool
    unknown: bool
    incomplete: bool


def _source_evidence(
    *,
    source_kind: SemanticSourceKind,
    source_id: str,
    source_revision: int | None,
    source_status: SemanticSourceStatus,
    captured_context_id: str | None,
    context_registry: ContextRegistry,
    current_context_id: str,
) -> SemanticContextSourceEvidence:
    compatibility = context_registry.compatibility(
        captured_context_id, current_context_id
    )
    return SemanticContextSourceEvidence(
        source_kind=source_kind,
        source_id=source_id,
        source_revision=source_revision,
        source_status=source_status,
        captured_context_id=captured_context_id,
        relation=compatibility.relation,
        compatibility_score=compatibility.score,
    )


def project_semantic_revision_context(
    revision: SemanticRevision,
    context_registry: ContextRegistry,
    current_context_id: str,
) -> SemanticContextProjection:
    """Project authoritative new-format provenance without mutating any authority."""

    if not isinstance(revision, SemanticRevision):
        raise TypeError("revision must be SemanticRevision")
    if not isinstance(context_registry, ContextRegistry):
        raise TypeError("context_registry must be ContextRegistry")
    validate_identifier(current_context_id)

    evidence = tuple(
        _source_evidence(
            source_kind=edge.source_kind,
            source_id=edge.source_id,
            source_revision=edge.source_revision,
            source_status=edge.source_status,
            captured_context_id=edge.captured_context_id,
            context_registry=context_registry,
            current_context_id=current_context_id,
        )
        for edge in revision.source_edges
    )
    if evidence:
        aggregate = sum(item.compatibility_score for item in evidence) / len(evidence)
    else:
        aggregate = context_registry.compatibility(None, current_context_id).score
    known_contexts = {
        edge.captured_context_id
        for edge in revision.source_edges
        if edge.captured_context_id is not None
    }
    return SemanticContextProjection(
        semantic_id=revision.semantic_id,
        provenance_classification=revision.provenance.classification,
        source_count=revision.provenance.source_count,
        source_evidence=evidence,
        aggregate_compatibility=aggregate,
        cross_context=len(known_contexts) > 1,
        unknown=(
            revision.provenance.unknown_source_count > 0
            or revision.provenance.source_count == 0
        ),
        incomplete=(
            revision.provenance.classification
            is SemanticProvenanceClass.INCOMPLETE
        ),
    )


def project_legacy_semantic_context(
    record: SemanticMemoryRecord,
    context_registry: ContextRegistry,
    current_context_id: str,
) -> SemanticContextProjection:
    """Project retained R09 evidence without DB1 lookup or durable backfill."""

    if not isinstance(record, SemanticMemoryRecord):
        raise TypeError("record must be SemanticMemoryRecord")
    if not isinstance(context_registry, ContextRegistry):
        raise TypeError("context_registry must be ContextRegistry")
    validate_identifier(record.id)
    validate_identifier(current_context_id)

    unique_source_ids: list[str] = []
    seen: set[str] = set()
    for source_id in record.source_episode_ids:
        checked = validate_identifier(source_id)
        if checked not in seen:
            seen.add(checked)
            unique_source_ids.append(checked)
    unique_source_ids.sort()

    retained_context_id = (
        None
        if record.context_id is None
        else validate_identifier(record.context_id)
    )
    evidence = tuple(
        _source_evidence(
            source_kind=SemanticSourceKind.EPISODIC,
            source_id=source_id,
            source_revision=None,
            source_status=SemanticSourceStatus.UNKNOWN,
            captured_context_id=retained_context_id,
            context_registry=context_registry,
            current_context_id=current_context_id,
        )
        for source_id in unique_source_ids
    )
    if evidence:
        aggregate = sum(item.compatibility_score for item in evidence) / len(evidence)
    else:
        aggregate = context_registry.compatibility(None, current_context_id).score

    classification = (
        SemanticProvenanceClass.SINGLE_CONTEXT
        if unique_source_ids and retained_context_id is not None
        else SemanticProvenanceClass.UNKNOWN
    )
    return SemanticContextProjection(
        semantic_id=record.id,
        provenance_classification=classification,
        source_count=len(unique_source_ids),
        source_evidence=evidence,
        aggregate_compatibility=aggregate,
        cross_context=False,
        unknown=retained_context_id is None,
        incomplete=False,
    )


class MemoryWorkingMemoryResolver:
    """Resolve Working Memory references using only committed domain reads."""

    def __init__(self, memory: DualMemorySystem) -> None:
        self._memory = memory

    def __call__(self, item: WorkingMemoryItem) -> WorkingMemoryResolution:
        return self.resolve(item)

    def resolve(self, item: WorkingMemoryItem) -> WorkingMemoryResolution:
        if not isinstance(item, WorkingMemoryItem):
            raise TypeError("item must be WorkingMemoryItem")

        # Importing the domain exceptions here keeps the bridge out of the
        # memory package's public initializer and avoids an import cycle.
        from suzka.memory.dual_memory_system import (
            EpisodicMemoryFormatError,
            EpisodicMemoryReadError,
            SemanticMemoryFormatError,
            SemanticMemoryReadError,
        )

        if item.source_kind is WorkingMemorySourceKind.EPISODIC:
            try:
                committed = self._memory.get_committed_episodic(item.source_id)
            except EpisodicMemoryReadError:
                return WorkingMemoryResolution(
                    WorkingMemoryResolutionStatus.UNAVAILABLE
                )
            except EpisodicMemoryFormatError:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)
            if committed is None:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MISSING)
            if committed.record.archived:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.ARCHIVED)
            return WorkingMemoryResolution(
                WorkingMemoryResolutionStatus.RESOLVED,
                committed.document,
                getattr(committed.record, "context_id", None),
            )

        if item.source_kind is WorkingMemorySourceKind.SEMANTIC:
            try:
                semantic = self._memory.get_committed_semantic(item.source_id)
            except SemanticMemoryReadError:
                return WorkingMemoryResolution(
                    WorkingMemoryResolutionStatus.UNAVAILABLE
                )
            except SemanticMemoryFormatError:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)
            if semantic is None:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MISSING)
            return WorkingMemoryResolution(
                WorkingMemoryResolutionStatus.RESOLVED,
                semantic.document,
                getattr(semantic.record, "context_id", None),
            )

        raise TypeError("Working Memory source kind is unsupported")

    def resolve_contextual(
        self,
        item: WorkingMemoryItem,
        context_registry: ContextRegistry,
        current_context_id: str,
    ) -> WorkingMemoryResolution:
        """Resolve Semantic Context compatibility without mutating Memory or Context."""

        if not isinstance(item, WorkingMemoryItem):
            raise TypeError("item must be WorkingMemoryItem")
        if item.source_kind is not WorkingMemorySourceKind.SEMANTIC:
            return self.resolve(item)

        from suzka.memory.dual_memory_system import (
            SEMANTIC_PROJECTION_SCHEMA,
            SemanticMemoryFormatError,
            SemanticMemoryReadError,
        )
        from suzka.memory.semantic_store import (
            SemanticStoreError,
            SemanticStoreUnavailable,
        )

        try:
            semantic = self._memory.get_committed_semantic(item.source_id)
        except SemanticMemoryReadError:
            return WorkingMemoryResolution(WorkingMemoryResolutionStatus.UNAVAILABLE)
        except SemanticMemoryFormatError:
            return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)
        if semantic is None:
            return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MISSING)

        if semantic.metadata.get("semantic_projection_schema") == SEMANTIC_PROJECTION_SCHEMA:
            try:
                stored = self._memory.semantic_store.load_current(item.source_id)
            except SemanticStoreUnavailable:
                return WorkingMemoryResolution(
                    WorkingMemoryResolutionStatus.UNAVAILABLE
                )
            except SemanticStoreError:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)
            if stored is None:
                return WorkingMemoryResolution(
                    WorkingMemoryResolutionStatus.UNAVAILABLE
                )
            projection = project_semantic_revision_context(
                stored.revision,
                context_registry,
                current_context_id,
            )
        else:
            try:
                projection = project_legacy_semantic_context(
                    semantic.record,
                    context_registry,
                    current_context_id,
                )
            except (TypeError, ValueError):
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)

        return WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED,
            semantic.document,
            context_projection=projection,
        )


__all__ = [
    "MemoryWorkingMemoryResolver",
    "SemanticContextProjection",
    "SemanticContextSourceEvidence",
    "project_legacy_semantic_context",
    "project_semantic_revision_context",
]
