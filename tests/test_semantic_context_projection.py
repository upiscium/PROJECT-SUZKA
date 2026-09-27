"""Focused U4 multi-Context Semantic projection contract tests."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from suzka.memory.memory_schema import MemoryRecordType, SemanticMemoryRecord
from suzka.memory.semantic_lifecycle import (
    SEMANTIC_MAX_SOURCE_EDGES,
    SemanticProvenanceClass,
    SemanticRevision,
    SemanticSourceEdge,
    SemanticSourceKind,
    SemanticSourceStatus,
    semantic_content_digest,
)
from suzka.memory.working_memory_resolver import (
    MemoryWorkingMemoryResolver,
    SemanticContextProjection,
    project_legacy_semantic_context,
    project_semantic_revision_context,
)
from suzka.runtime.context import (
    ContextRelation,
    ContextRegistry,
    ContextType,
)
from suzka.runtime.working_memory import (
    WorkingMemory,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemorySourceKind,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _registry() -> ContextRegistry:
    registry = ContextRegistry(clock=lambda: NOW)
    for context_id in ("context-current", "context-y", "context-related"):
        registry.create(context_id, ContextType.CONVERSATION, "chat")
    registry.relate("context-current", "context-related")
    return registry


def _edge(
    source_id: str,
    context_id: str | None,
    *,
    status: SemanticSourceStatus = SemanticSourceStatus.AVAILABLE,
) -> SemanticSourceEdge:
    return SemanticSourceEdge(
        source_kind=SemanticSourceKind.EPISODIC,
        source_id=source_id,
        captured_context_id=context_id,
        source_status=status,
    )


def _revision(
    semantic_id: str,
    edges: tuple[SemanticSourceEdge, ...],
    *,
    text: str = "fact",
) -> SemanticRevision:
    return SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content=text,
        content_digest=semantic_content_digest(text),
        created_at=NOW,
        source_edges=edges,
    )


def _legacy(
    semantic_id: str,
    source_ids: list[str],
    context_id: str | None,
) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        id=semantic_id,
        text="legacy fact",
        source_episode_ids=source_ids,
        record_type=MemoryRecordType.SEMANTIC_MEMORY,
        created_at=NOW.isoformat(),
        context_id=context_id,
    )


def test_new_format_projection_distinguishes_context_shapes_and_uses_mean() -> None:
    registry = _registry()
    cases = (
        (
            _revision(
                "semantic-xx",
                (
                    _edge("episode-a", "context-current"),
                    _edge("episode-b", "context-current"),
                ),
            ),
            SemanticProvenanceClass.SINGLE_CONTEXT,
            1.0,
            False,
            False,
        ),
        (
            _revision(
                "semantic-xy",
                (
                    _edge("episode-a", "context-current"),
                    _edge("episode-b", "context-y"),
                ),
            ),
            SemanticProvenanceClass.MULTI_CONTEXT,
            0.6,
            True,
            False,
        ),
        (
            _revision(
                "semantic-x-unknown",
                (
                    _edge("episode-a", "context-current"),
                    _edge("episode-b", None),
                ),
            ),
            SemanticProvenanceClass.INCOMPLETE,
            0.725,
            False,
            True,
        ),
        (
            _revision(
                "semantic-all-unknown",
                (
                    _edge(
                        "episode-a",
                        None,
                        status=SemanticSourceStatus.UNKNOWN,
                    ),
                ),
            ),
            SemanticProvenanceClass.UNKNOWN,
            0.45,
            False,
            True,
        ),
        (
            _revision("semantic-source-less", ()),
            SemanticProvenanceClass.UNKNOWN,
            0.45,
            False,
            True,
        ),
    )

    for revision, classification, expected_mean, cross_context, unknown in cases:
        projection = project_semantic_revision_context(
            revision, registry, "context-current"
        )
        assert projection.provenance_classification is classification
        assert projection.source_count == len(revision.source_edges)
        assert projection.aggregate_compatibility == pytest.approx(expected_mean)
        assert projection.cross_context is cross_context
        assert projection.unknown is unknown

    source_less = project_semantic_revision_context(
        cases[-1][0], registry, "context-current"
    )
    assert source_less.source_count == 0
    assert source_less.source_evidence == ()


def test_projection_is_canonical_for_duplicate_order_and_preserves_status() -> None:
    registry = _registry()
    first = _edge("episode-a", "context-current")
    second = _edge(
        "episode-b",
        "context-related",
        status=SemanticSourceStatus.RETRACTED,
    )
    left = _revision("semantic-left", (second, first, first))
    right = _revision("semantic-right", (first, second))

    left_projection = project_semantic_revision_context(
        left, registry, "context-current"
    )
    right_projection = project_semantic_revision_context(
        right, registry, "context-current"
    )

    assert left_projection.source_count == right_projection.source_count == 2
    assert tuple(
        (
            evidence.source_id,
            evidence.source_status,
            evidence.relation,
            evidence.compatibility_score,
        )
        for evidence in left_projection.source_evidence
    ) == tuple(
        (
            evidence.source_id,
            evidence.source_status,
            evidence.relation,
            evidence.compatibility_score,
        )
        for evidence in right_projection.source_evidence
    )
    retracted = next(
        evidence
        for evidence in left_projection.source_evidence
        if evidence.source_id == "episode-b"
    )
    assert retracted.source_status is SemanticSourceStatus.RETRACTED
    assert retracted.relation is ContextRelation.RELATED
    assert left_projection.incomplete is True


def test_missing_retained_context_becomes_unknown_context_without_revision_change() -> None:
    registry = _registry()
    revision = _revision(
        "semantic-missing-context",
        (_edge("episode-a", "context-rolled-back"),),
    )
    before = revision.revision_digest
    registry_before = registry.state

    first = project_semantic_revision_context(
        revision, registry, "context-current"
    )
    second = project_semantic_revision_context(
        revision, registry, "context-current"
    )

    assert first == second
    assert first.source_evidence[0].captured_context_id == "context-rolled-back"
    assert first.source_evidence[0].relation is ContextRelation.UNKNOWN_CONTEXT
    assert first.source_evidence[0].compatibility_score == pytest.approx(0.35)
    assert revision.revision_digest == before
    assert registry.state == registry_before


def test_legacy_bridge_uses_only_retained_evidence_and_deduplicates_sources() -> None:
    registry = _registry()
    known = project_legacy_semantic_context(
        _legacy(
            "semantic-known",
            ["episode-b", "episode-a", "episode-a"],
            "context-current",
        ),
        registry,
        "context-current",
    )
    unknown = project_legacy_semantic_context(
        _legacy(
            "semantic-unknown",
            ["episode-b", "episode-a"],
            None,
        ),
        registry,
        "context-current",
    )
    source_less = project_legacy_semantic_context(
        _legacy("semantic-source-less", [], None),
        registry,
        "context-current",
    )

    assert known.provenance_classification is SemanticProvenanceClass.SINGLE_CONTEXT
    assert known.source_count == 2
    assert [evidence.source_id for evidence in known.source_evidence] == [
        "episode-a",
        "episode-b",
    ]
    assert known.aggregate_compatibility == pytest.approx(1.0)

    assert unknown.provenance_classification is SemanticProvenanceClass.UNKNOWN
    assert unknown.source_count == 2
    assert all(
        evidence.relation is ContextRelation.LEGACY_UNKNOWN
        for evidence in unknown.source_evidence
    )
    assert unknown.aggregate_compatibility == pytest.approx(0.45)

    assert source_less.provenance_classification is SemanticProvenanceClass.UNKNOWN
    assert source_less.source_count == 0
    assert source_less.source_evidence == ()
    assert source_less.aggregate_compatibility == pytest.approx(0.45)


def test_legacy_contextual_resolver_never_reads_db1() -> None:
    registry = _registry()
    record = _legacy(
        "semantic-legacy",
        ["episode-a", "episode-b"],
        "context-current",
    )

    class Memory:
        def get_committed_semantic(self, source_id: str) -> object:
            assert source_id == record.id
            return SimpleNamespace(
                document=record.text,
                metadata={},
                record=record,
            )

        def get_committed_episodic(self, _source_id: str) -> object:
            raise AssertionError("legacy Context projection must not read DB1")

    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    item = working.admit(
        WorkingMemorySourceKind.SEMANTIC,
        record.id,
        activation=1.0,
        salience=1.0,
    ).item
    resolver = MemoryWorkingMemoryResolver(Memory())  # type: ignore[arg-type]

    resolution = resolver.resolve_contextual(
        item,
        registry,
        "context-current",
    )

    assert resolution.status is WorkingMemoryResolutionStatus.RESOLVED
    assert resolution.source_context_id == "context-current"
    assert isinstance(resolution.context_projection, SemanticContextProjection)
    assert resolution.context_projection.aggregate_compatibility == pytest.approx(1.0)


def test_new_format_contextual_resolver_uses_authoritative_revision() -> None:
    registry = _registry()
    revision = _revision(
        "semantic-new",
        (
            _edge("episode-a", "context-current"),
            _edge("episode-b", "context-y"),
        ),
    )

    class Store:
        def load_current(self, semantic_id: str) -> object:
            assert semantic_id == revision.semantic_id
            return SimpleNamespace(revision=revision)

    class Memory:
        semantic_store = Store()

        def get_committed_semantic(self, source_id: str) -> object:
            assert source_id == revision.semantic_id
            return SimpleNamespace(
                document=revision.semantic_content,
                metadata={"semantic_projection_schema": "r12.semantic.v1"},
                record=_legacy(revision.semantic_id, ["episode-a"], None),
            )

    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    item = working.admit(
        WorkingMemorySourceKind.SEMANTIC,
        revision.semantic_id,
        activation=1.0,
        salience=1.0,
    ).item
    resolution = MemoryWorkingMemoryResolver(Memory()).resolve_contextual(  # type: ignore[arg-type]
        item,
        registry,
        "context-current",
    )

    assert resolution.status is WorkingMemoryResolutionStatus.RESOLVED
    assert isinstance(resolution.context_projection, SemanticContextProjection)
    assert resolution.context_projection.provenance_classification is (
        SemanticProvenanceClass.MULTI_CONTEXT
    )
    assert resolution.context_projection.aggregate_compatibility == pytest.approx(0.6)


def test_contextual_working_memory_uses_semantic_mean_not_best_source() -> None:
    registry = _registry()
    projection_mean = project_semantic_revision_context(
        _revision(
            "semantic-mean",
            (
                _edge("episode-a", "context-current"),
                _edge("episode-b", "context-y"),
            ),
        ),
        registry,
        "context-current",
    )
    projection_related = project_semantic_revision_context(
        _revision(
            "semantic-related",
            (
                _edge("episode-c", "context-related"),
                _edge("episode-d", "context-related"),
            ),
        ),
        registry,
        "context-current",
    )
    assert projection_mean.aggregate_compatibility == pytest.approx(0.6)
    assert projection_related.aggregate_compatibility == pytest.approx(0.75)

    working = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    working.admit(
        WorkingMemorySourceKind.SEMANTIC,
        "semantic-mean",
        activation=1.0,
        salience=1.0,
    )
    working.admit(
        WorkingMemorySourceKind.SEMANTIC,
        "semantic-related",
        activation=0.9,
        salience=0.9,
    )

    class Resolver:
        def __call__(self, _item: object) -> None:
            raise AssertionError("contextual resolver path must be used")

        def resolve_contextual(
            self,
            item: object,
            _registry: ContextRegistry,
            _current_context_id: str,
        ) -> WorkingMemoryResolution:
            source_id = getattr(item, "source_id")
            projection = (
                projection_mean
                if source_id == "semantic-mean"
                else projection_related
            )
            return WorkingMemoryResolution(
                WorkingMemoryResolutionStatus.RESOLVED,
                source_id,
                context_projection=projection,
            )

    before = (working.revision, working.items, registry.state)
    view = working.select_contextual(
        Resolver(),  # type: ignore[arg-type]
        registry,
        "context-current",
    )

    assert [selection.source_id for selection in view.selected] == [
        "semantic-related",
        "semantic-mean",
    ]
    assert view.selected[0].effective_score == pytest.approx(0.675)
    assert view.selected[1].effective_score == pytest.approx(0.6)
    assert view.selected[1].context_relation is None
    assert view.selected[1].context_projection is projection_mean
    assert (working.revision, working.items, registry.state) == before




def test_legacy_projection_overflow_fails_closed() -> None:
    registry = _registry()
    oversized = _legacy(
        "semantic-overflow",
        [
            f"episode-{index}"
            for index in range(SEMANTIC_MAX_SOURCE_EDGES + 1)
        ],
        None,
    )

    with pytest.raises(ValueError, match="exceeds"):
        project_legacy_semantic_context(
            oversized,
            registry,
            "context-current",
        )

def test_nonresolved_working_memory_cannot_carry_context_projection() -> None:
    registry = _registry()
    projection = project_semantic_revision_context(
        _revision("semantic-private", ()),
        registry,
        "context-current",
    )

    with pytest.raises(ValueError):
        WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.MISSING,
            context_projection=projection,
        )
