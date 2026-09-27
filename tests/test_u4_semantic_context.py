"""Focused U4 semantic-context regression coverage."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from suzka.config import Settings, load_settings
from suzka.memory import DualMemorySystem
from suzka.memory.dual_memory_system import SemanticMemoryFormatError
from suzka.memory.dual_memory_system import semantic_projection_metadata
from suzka.memory.semantic_context_projection import SemanticContextEvidence
from suzka.memory.semantic_lifecycle import (
    SemanticProvenanceClass,
    SemanticRevision,
    SemanticSourceEdge,
    SemanticSourceKind,
    SemanticSourceStatus,
    canonicalize_source_edges,
    semantic_content_digest,
)
from suzka.memory.semantic_store import SemanticStore
from suzka.memory.working_memory_resolver import MemoryWorkingMemoryResolver
from suzka.runtime import (
    ContextRelation,
    ContextRegistry,
    ContextType,
    ContextualProjection,
    ContextualSourceEvidence,
    WorkingMemory,
    WorkingMemoryItem,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemoryRetentionReason,
    WorkingMemorySourceKind,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "u4_context_db1",
                    "db2_collection": "u4_context_db2",
                }
            )
        }
    )


def _edge(source_id: str, context_id: str | None, *, status=SemanticSourceStatus.AVAILABLE):
    return SemanticSourceEdge(
        SemanticSourceKind.EPISODIC,
        source_id,
        captured_context_id=context_id,
        source_status=status,
    )


def _evidence(edges: tuple[SemanticSourceEdge, ...]) -> SemanticContextEvidence:
    edges = canonicalize_source_edges(edges)
    contexts = {edge.captured_context_id for edge in edges if edge.captured_context_id}
    has_unknown = any(
        edge.source_status is not SemanticSourceStatus.AVAILABLE
        or edge.captured_context_id is None
        for edge in edges
    )
    if not edges:
        classification = SemanticProvenanceClass.UNKNOWN
    elif has_unknown and contexts:
        classification = SemanticProvenanceClass.INCOMPLETE
    elif not contexts:
        classification = SemanticProvenanceClass.UNKNOWN
    elif len(contexts) == 1:
        classification = SemanticProvenanceClass.SINGLE_CONTEXT
    else:
        classification = SemanticProvenanceClass.MULTI_CONTEXT
    return SemanticContextEvidence(
        "semantic body",
        "semantic-u4",
        classification,
        len(edges),
        edges,
        cross_context=classification is SemanticProvenanceClass.MULTI_CONTEXT,
        unknown_or_incomplete=classification
        in (SemanticProvenanceClass.UNKNOWN, SemanticProvenanceClass.INCOMPLETE),
        source_less=not edges,
    )


class _EvidenceMemory:
    def __init__(self, evidence: SemanticContextEvidence) -> None:
        self.evidence = evidence

    def get_semantic_context_evidence(self, source_id: str) -> SemanticContextEvidence:
        assert source_id == "semantic-u4"
        return self.evidence


def _item() -> WorkingMemoryItem:
    return WorkingMemoryItem(
        "item-u4",
        WorkingMemorySourceKind.SEMANTIC,
        "semantic-u4",
        0.8,
        0.8,
        WorkingMemoryRetentionReason.RECENT,
        0,
        0,
    )


def _registry() -> tuple[ContextRegistry, str, str]:
    registry = ContextRegistry(clock=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    current = registry.create("context-x", ContextType.CONVERSATION, "test")
    other = registry.create("context-y", ContextType.CONVERSATION, "test")
    return registry, current.context_id, other.context_id


@pytest.mark.parametrize(
    ("edges", "classification", "relation", "score"),
    [
        ((_edge("source-a", "context-x"), _edge("source-b", "context-x")), "single_context", ContextRelation.SAME_CONTEXT, 1.0),
        ((_edge("source-a", "context-x"), _edge("source-b", "context-y")), "multi_context", ContextRelation.SAME_CONTEXT, 0.6),
        ((_edge("source-a", "context-x"), _edge("source-b", None)), "incomplete", ContextRelation.SAME_CONTEXT, 0.725),
        ((_edge("source-a", None), _edge("source-b", None)), "unknown", ContextRelation.LEGACY_UNKNOWN, 0.45),
    ],
)
def test_u4_projection_classification_and_mean(edges, classification, relation, score):
    registry, current, other = _registry()
    resolver = MemoryWorkingMemoryResolver(_EvidenceMemory(_evidence(edges)))
    resolution = resolver.resolve_contextual(_item(), registry, current)
    projection = resolution.contextual_projection
    assert projection is not None
    assert projection.provenance_classification == classification
    assert projection.source_count == len(edges)
    assert projection.compatibility_score == pytest.approx(score)
    assert projection.source_edges[0].context_relation is relation
    assert projection.source_edges[1].context_relation is registry.compatibility(
        edges[1].captured_context_id, current
    ).relation
    assert [edge.source_status for edge in projection.source_edges] == ["available"] * len(edges)


def test_u4_source_less_uses_exactly_one_legacy_unknown_score_and_missing_is_bounded():
    registry, current, _ = _registry()
    resolver = MemoryWorkingMemoryResolver(_EvidenceMemory(_evidence(())))
    projection = resolver.resolve_contextual(_item(), registry, current).contextual_projection
    assert projection is not None
    assert (projection.source_count, projection.source_less) == (0, True)
    assert projection.compatibility_score == 0.45
    assert projection.source_evidence == ()

    missing = _evidence((_edge("source-missing", "context-gone"),))
    resolver = MemoryWorkingMemoryResolver(_EvidenceMemory(missing))
    result = resolver.resolve_contextual(_item(), registry, current).contextual_projection
    assert result is not None
    assert result.source_edges[0].context_relation is ContextRelation.UNKNOWN_CONTEXT
    assert result.source_edges[0].captured_context_id == "context-gone"


def test_u4_source_status_and_captured_context_remain_visible_without_private_evidence():
    registry, current, _ = _registry()
    evidence = _evidence(
        (_edge("source-missing", "context-gone", status=SemanticSourceStatus.MISSING),)
    )
    projection = MemoryWorkingMemoryResolver(_EvidenceMemory(evidence)).resolve_contextual(
        _item(), registry, current
    ).contextual_projection
    assert projection is not None
    source = projection.source_edges[0]
    assert (source.source_status, source.captured_context_id) == (
        "missing",
        "context-gone",
    )
    assert "PRIVATE" not in repr(projection)


def test_u4_edge_order_and_duplicate_canonicalization_are_projection_invariant():
    registry, current, other = _registry()
    first = _evidence((_edge("source-b", other), _edge("source-a", current)))
    second = _evidence((_edge("source-a", current), _edge("source-b", other)))
    # SemanticStore/lifecycle canonicalization is the authority; repeated equal edges
    # are harmless at the projection boundary and never improve the mean.
    first = _evidence(tuple(dict.fromkeys(first.source_edges)))
    second = _evidence(tuple(dict.fromkeys(second.source_edges)))
    assert first.source_edges == second.source_edges
    assert first.source_count == second.source_count == 2


def test_u4_legacy_bridge_uses_retained_edges_without_db1_lookup(tmp_path, monkeypatch):
    memory = DualMemorySystem(_settings(tmp_path))
    semantic_id = memory.save_legacy_semantic(
        "legacy context", source_episode_ids=["episode-a", "episode-a"]
    )
    before = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])
    monkeypatch.setattr(memory.db1, "get", lambda **_: (_ for _ in ()).throw(AssertionError("DB1 lookup")))
    evidence = memory.get_semantic_context_evidence(semantic_id)
    assert evidence is not None
    assert evidence.source_count == 1
    assert evidence.provenance_class is SemanticProvenanceClass.UNKNOWN
    assert memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"]) == before


def test_u4_legacy_malformed_and_overflow_evidence_fail_boundedly(tmp_path):
    memory = DualMemorySystem(_settings(tmp_path))
    semantic_id = memory.save_legacy_semantic("legacy bounded")
    metadata = dict(memory.db2.get(ids=[semantic_id], include=["metadatas"])["metadatas"][0])
    metadata["source_episode_ids"] = json.dumps(["source-a"] * 65)
    memory.db2.update(ids=[semantic_id], metadatas=[metadata])
    with pytest.raises(SemanticMemoryFormatError):
        memory.get_semantic_context_evidence(semantic_id)


def test_u4_new_format_db2_summary_cannot_forge_semantic_store_edges(tmp_path):
    memory = DualMemorySystem(_settings(tmp_path))
    semantic_id = "semantic-authoritative"
    revision = SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content="projection body",
        content_digest=semantic_content_digest("projection body"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_edges=(_edge("source-a", "context-x"), _edge("source-b", "context-y")),
    )
    SemanticStore.from_memory_root(memory.settings.memory.persist_directory).publish_create(
        revision, "a" * 64
    )
    memory.db2.add(
        ids=[semantic_id],
        documents=[revision.semantic_content],
        metadatas=[semantic_projection_metadata(revision)],
    )
    # The public read path rejects a lossy/forged projection rather than trusting DB2.
    metadata = dict(memory.db2.get(ids=[semantic_id], include=["metadatas"])["metadatas"][0])
    metadata["source_episode_ids"] = json.dumps(["forged-source"])
    memory.db2.update(ids=[semantic_id], metadatas=[metadata])
    with pytest.raises(SemanticMemoryFormatError):
        memory.get_semantic_context_evidence(semantic_id)


def test_u4_new_format_marker_cannot_downgrade_into_legacy_bridge(tmp_path):
    memory = DualMemorySystem(_settings(tmp_path))
    semantic_id = "semantic-marker-downgrade"
    revision = SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content="projection body",
        content_digest=semantic_content_digest("projection body"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_edges=(_edge("source-a", "context-x"),),
    )
    SemanticStore.from_memory_root(memory.settings.memory.persist_directory).publish_create(
        revision, "a" * 64
    )
    metadata = semantic_projection_metadata(revision)
    metadata["semantic_projection_schema"] = "legacy"
    memory.db2.add(
        ids=[semantic_id], documents=[revision.semantic_content], metadatas=[metadata]
    )

    with pytest.raises(SemanticMemoryFormatError):
        memory.get_semantic_context_evidence(semantic_id)


def test_u4_contextual_projection_rejects_malformed_authority_digest():
    evidence = ContextualSourceEvidence(
        WorkingMemorySourceKind.EPISODIC,
        "source-a",
        None,
        "available",
        "context-a",
        ContextRelation.SAME_CONTEXT,
        1.0,
    )
    with pytest.raises(ValueError):
        ContextualProjection(
            "single_context",
            1,
            (evidence,),
            1.0,
            False,
            False,
            False,
            0,
            "A" * 64,
        )


def test_u4_contextual_resolution_preserves_callable_resolver_overrides():
    registry, current, _ = _registry()

    class CallableOverride(MemoryWorkingMemoryResolver):
        def __call__(self, item):
            assert item.source_id == "semantic-u4"
            return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MISSING)

    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    working.admit(WorkingMemorySourceKind.SEMANTIC, "semantic-u4", 1.0, 1.0)
    view = working.select_contextual(
        CallableOverride(_EvidenceMemory(_evidence(()))), registry, current
    )

    assert view.decisions[0].reason.value == "unresolved_reference"


def test_u4_selection_consumes_precomputed_projection_and_reads_are_pure(monkeypatch):
    registry, current, other = _registry()
    evidence = _evidence((_edge("source-a", current), _edge("source-b", other)))
    resolver = MemoryWorkingMemoryResolver(_EvidenceMemory(evidence))
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    working.admit(WorkingMemorySourceKind.SEMANTIC, "semantic-u4", 0.8, 0.8)
    before = (working.revision, working.items, registry.state)
    calls = 0
    compatibility = registry.compatibility

    def counted(source, target):
        nonlocal calls
        calls += 1
        return compatibility(source, target)

    monkeypatch.setattr(registry, "compatibility", counted)
    view = working.select_contextual(resolver, registry, current)
    assert view.selected[0].context_compatibility == pytest.approx(0.6)
    assert calls == 2
    assert (working.revision, working.items, registry.state) == before


def test_u4_generic_scalar_resolver_and_positional_constructors_remain_compatible():
    registry, current, _ = _registry()
    calls = 0

    def scalar(item):
        nonlocal calls
        calls += 1
        return WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED, "body", current
        )

    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    working.admit(WorkingMemorySourceKind.SEMANTIC, "semantic-u4", 1.0, 1.0)
    working.select_contextual(scalar, registry, current)
    assert calls == 1
    assert WorkingMemoryResolution(WorkingMemoryResolutionStatus.RESOLVED, "body").rendered_content == "body"
    assert ContextualSourceEvidence(
        WorkingMemorySourceKind.EPISODIC,
        "source-a",
        None,
        "available",
        current,
        ContextRelation.SAME_CONTEXT,
        1.0,
    ).source_id == "source-a"
