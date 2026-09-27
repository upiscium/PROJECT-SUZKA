"""Bridge committed dual-memory reads into the runtime resolution contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from suzka.runtime.working_memory import (
    ContextualProjection,
    ContextualSourceEvidence,
    WorkingMemoryItem,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemorySourceKind,
)
from suzka.runtime.context import ContextRegistry

if TYPE_CHECKING:
    from suzka.memory.dual_memory_system import DualMemorySystem


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
        """Resolve content and attach request-scoped compatibility evidence."""

        from suzka.memory.dual_memory_system import (
            SemanticMemoryFormatError,
            SemanticMemoryReadError,
        )

        # Contextual selection must still honor an injected/subclassed scalar
        # resolver.  This keeps the pre-U4 resolver contract authoritative for
        # bounded failure outcomes and custom source adapters.
        bound_resolve = getattr(self.resolve, "__func__", None)
        bound_call = getattr(self.__call__, "__func__", None)
        if (
            bound_resolve is not MemoryWorkingMemoryResolver.resolve
            or bound_call is not MemoryWorkingMemoryResolver.__call__
        ):
            return self(item)

        if item.source_kind is not WorkingMemorySourceKind.SEMANTIC:
            # Episodic resolution retains the pre-U4 scalar path.  The
            # contextual selector computes its existing one-context score.
            return self.resolve(item)

        try:
            semantic = self._memory.get_semantic_context_evidence(item.source_id)
        except SemanticMemoryReadError:
            return WorkingMemoryResolution(WorkingMemoryResolutionStatus.UNAVAILABLE)
        except SemanticMemoryFormatError:
            return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)
        if semantic is None:
            return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MISSING)

        source_evidence: list[ContextualSourceEvidence] = []
        for edge in semantic.source_edges:
            compatibility = context_registry.compatibility(
                edge.captured_context_id, current_context_id
            )
            source_evidence.append(
                ContextualSourceEvidence(
                    edge.source_kind.value,
                    edge.source_id,
                    edge.source_revision,
                    edge.source_status.value,
                    edge.captured_context_id,
                    compatibility.relation,
                    compatibility.score,
                )
            )
        if not source_evidence:
            compatibility = context_registry.compatibility(None, current_context_id)
        aggregate = (
            sum(e.compatibility_score for e in source_evidence) / len(source_evidence)
            if source_evidence
            else compatibility.score
        )
        projection = ContextualProjection(
            semantic.provenance_class.value,
            semantic.source_count,
            tuple(source_evidence),
            aggregate,
            semantic.cross_context,
            semantic.unknown_or_incomplete,
            semantic.source_less,
            semantic.authoritative_revision,
            semantic.authoritative_digest,
        )
        return WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED,
            semantic.document,
            None,
            projection,
        )
