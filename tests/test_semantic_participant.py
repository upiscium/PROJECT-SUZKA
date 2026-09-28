"""R07 Semantic batch participant and projection-boundary tests."""

import os
import stat
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from suzka.config import Settings, load_settings
from suzka.memory import (
    DualMemorySystem,
    EpisodicMemoryReadError,
    MemorySemanticParticipant,
    SemanticBatchEntry,
    SemanticBatchOperation,
    SemanticCreateIntent,
    SemanticRevisionIntent,
    SemanticStore,
    semantic_id_for_batch_entry,
)
from suzka.memory.semantic_store import SemanticStoreUnavailable
from suzka.memory.semantic_lifecycle import (
    SemanticLifecycle,
    SemanticRevision,
    SemanticRevisionOperation,
    SemanticRevisionReason,
    SemanticSourceEdge,
    SemanticSourceKind,
    SemanticSourceStatus,
    semantic_content_digest,
)
from suzka.runtime import (
    AbortOutcome,
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    ParticipantDivergedError,
    ParticipantOutcome,
    ParticipantUnavailableError,
    StartupParticipantOutcome,
    TransactionBinding,
    TransactionCoordinator,
    TransactionKind,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "semantic_participant_db1",
                    "db2_collection": "semantic_participant_db2",
                }
            )
        }
    )


def _event(sequence: int = 1) -> AgentEvent:
    return AgentEvent(
        event_id=f"11111111-1111-4111-8111-{sequence:012d}",
        event_type=AgentEventType.SLEEP,
        source=AgentEventSource.API_SLEEP_RUN,
        requested_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=sequence),
        processing_sequence=sequence,
    )


def _create_participant(
    memory: DualMemorySystem,
    *,
    event: AgentEvent | None = None,
    source_edges: tuple[SemanticSourceEdge, ...] = (),
) -> tuple[MemorySemanticParticipant, TransactionBinding, SemanticRevision]:
    current_event = event or _event()
    transaction_id = TransactionCoordinator.derive_transaction_id(
        current_event, TransactionKind.EVENT_MUTATION
    )
    semantic_id = semantic_id_for_batch_entry(transaction_id, 0)
    revision = SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content="visible semantic fact",
        content_digest=semantic_content_digest("visible semantic fact"),
        created_at=current_event.requested_at,
        event_id=current_event.event_id,
        event_sequence=current_event.processing_sequence,
    )
    revision = replace(revision, source_edges=source_edges, provenance_class=None)
    operation = SemanticBatchOperation(
        transaction_id,
        (SemanticBatchEntry(0, SemanticCreateIntent(revision)),),
    )
    participant = MemorySemanticParticipant(
        memory,
        SemanticStore.from_memory_root(memory.settings.memory.persist_directory),
        operation,
    )
    binding = TransactionBinding(
        transaction_id,
        current_event.event_id,
        current_event.processing_sequence or 0,
        participant.participant_id,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    return participant, binding, revision


def test_deterministic_identity_and_bounded_batch_contract() -> None:
    transaction_id = "22222222-2222-4222-8222-222222222222"
    assert semantic_id_for_batch_entry(transaction_id, 0) == semantic_id_for_batch_entry(
        transaction_id, 0
    )
    with pytest.raises(ValueError):
        semantic_id_for_batch_entry(transaction_id, 128)

    with pytest.raises(ValueError, match="contiguous"):
        SemanticBatchOperation(
            transaction_id,
            (SemanticBatchEntry(1, SemanticCreateIntent(_revision_for_slot(transaction_id, 0, 1))),),
        )


def _revision_for_slot(transaction_id: str, index: int, sequence: int) -> SemanticRevision:
    semantic_id = semantic_id_for_batch_entry(transaction_id, index)
    content = f"slot {index}"
    return SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content=content,
        content_digest=semantic_content_digest(content),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        event_id=f"11111111-1111-4111-8111-{sequence:012d}",
        event_sequence=sequence,
    )


def test_prepare_is_non_authoritative_and_finalize_is_idempotent(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)

    participant.prepare(binding)

    assert participant.store.load_pending(binding.transaction_id) is not None
    assert participant.store.load_current(revision.semantic_id) is None
    assert memory.db2.get(ids=[revision.semantic_id], include=["documents", "metadatas"])["ids"] == []

    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    assert participant.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert participant.store.load_pending(binding.transaction_id) is None
    assert memory.get_committed_semantic(revision.semantic_id) is not None

    recovered = MemorySemanticParticipant.from_pending(
        memory,
        participant.store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=binding.event_id,
        processing_sequence=binding.processing_sequence,
    )
    assert recovered.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT

    participant.store.remove_receipt(binding.transaction_id)
    reconstructed = MemorySemanticParticipant.from_pending(
        memory,
        participant.store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=binding.event_id,
        processing_sequence=binding.processing_sequence,
    )
    assert reconstructed.operation == participant.operation


def test_mismatched_binding_transaction_fails_before_any_write(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    other_event = _event(2)
    other_transaction_id = TransactionCoordinator.derive_transaction_id(
        other_event, TransactionKind.EVENT_MUTATION
    )
    mismatched = replace(
        binding,
        transaction_id=other_transaction_id,
        event_id=other_event.event_id,
        processing_sequence=other_event.processing_sequence or 0,
    )

    with pytest.raises(ParticipantDivergedError, match="binding is invalid"):
        participant.prepare(mismatched)

    assert participant.store.load_pending(mismatched.transaction_id) is None
    assert participant.store.load_receipt(mismatched.transaction_id) is None
    assert participant.store.load_pending(binding.transaction_id) is None
    assert participant.store.load_receipt(binding.transaction_id) is None
    assert participant.store.load_current(revision.semantic_id) is None
    assert memory.db2.get(ids=[revision.semantic_id], include=["documents", "metadatas"])["ids"] == []


def test_pre_internal_abort_removes_pending_without_publication(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    participant.prepare(binding)

    assert participant.abort(binding) is AbortOutcome.ABORTED
    assert participant.store.load_current(revision.semantic_id) is None
    assert participant.store.load_receipt(binding.transaction_id) is None


def test_partial_lifecycle_publication_rolls_forward_without_duplicate_revision(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    participant.prepare(binding)
    participant.store.publish_create(revision, participant.operation_digest)

    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    assert participant.store.load_current(revision.semantic_id) is not None
    assert participant.store.load_receipt(binding.transaction_id) is not None


def test_partial_lifecycle_publication_restarts_from_pending_batch(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    event = _event()
    transaction_id = TransactionCoordinator.derive_transaction_id(
        event, TransactionKind.EVENT_MUTATION
    )
    revisions = tuple(
        _revision_for_slot(transaction_id, index, event.processing_sequence or 0)
        for index in range(2)
    )
    operation = SemanticBatchOperation(
        transaction_id,
        tuple(
            SemanticBatchEntry(index, SemanticCreateIntent(revision))
            for index, revision in enumerate(revisions)
        ),
    )
    store = SemanticStore.from_memory_root(memory.settings.memory.persist_directory)
    participant = MemorySemanticParticipant(memory, store, operation)
    binding = TransactionBinding(
        transaction_id,
        event.event_id,
        event.processing_sequence or 0,
        participant.participant_id,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    participant.prepare(binding)
    store.publish_create(revisions[0], participant.operation_digest, batch_index=0)

    restarted_memory = DualMemorySystem(_settings(tmp_path))
    restarted_store = SemanticStore.from_memory_root(
        restarted_memory.settings.memory.persist_directory
    )
    restarted = MemorySemanticParticipant.from_pending(
        restarted_memory,
        restarted_store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=binding.event_id,
        processing_sequence=binding.processing_sequence,
    )

    assert restarted.finalize(binding) is ParticipantOutcome.FINALIZED
    assert restarted.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert all(
        restarted_store.load_current(revision.semantic_id) is not None
        for revision in revisions
    )
    assert restarted_store.load_pending(binding.transaction_id) is None
    assert restarted_store.load_receipt(binding.transaction_id) is not None


@pytest.mark.parametrize("fault_stage", ["temp_fsync", "link", "post_link"])
def test_first_create_publication_crash_leaves_pending_and_restart_finalizes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_stage: str
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    participant.prepare(binding)

    import suzka.memory.semantic_store as semantic_store_module

    real_fsync = semantic_store_module.os.fsync
    real_link = semantic_store_module.os.link
    real_unlink = semantic_store_module.os.unlink
    crashed = False

    def fail_temp_fsync(descriptor: int) -> None:
        nonlocal crashed
        if (
            fault_stage == "temp_fsync"
            and stat.S_ISREG(os.fstat(descriptor).st_mode)
            and not crashed
        ):
            crashed = True
            raise OSError("simulated first-create publication crash")
        real_fsync(descriptor)

    def fail_link(
        source: str, destination: str, *args: object, **kwargs: object
    ) -> None:
        nonlocal crashed
        if fault_stage == "link" and not crashed:
            crashed = True
            raise OSError("simulated first-create link crash")
        real_link(source, destination, *args, **kwargs)

    def fail_unlink(path: object, *args: object, **kwargs: object) -> None:
        nonlocal crashed
        if (
            fault_stage == "post_link"
            and isinstance(path, str)
            and ".publish-" in path
            and not crashed
        ):
            crashed = True
            raise OSError("simulated first-create post-link crash")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(semantic_store_module.os, "fsync", fail_temp_fsync)
    monkeypatch.setattr(semantic_store_module.os, "link", fail_link)
    monkeypatch.setattr(semantic_store_module.os, "unlink", fail_unlink)
    with pytest.raises(ParticipantUnavailableError, match="atomic publication"):
        participant.finalize(binding)
    assert crashed
    record_directory = participant.store.records_root / revision.semantic_id
    assert record_directory.is_dir()
    if fault_stage == "post_link":
        assert (record_directory / "0.json").exists()
        assert any(".publish-" in item.name for item in record_directory.iterdir())
    else:
        assert tuple(record_directory.iterdir()) == ()
    assert participant.store.load_pending(binding.transaction_id) is not None

    monkeypatch.undo()
    monkeypatch.setattr(semantic_store_module.os, "fsync", real_fsync)
    restarted_memory = DualMemorySystem(_settings(tmp_path))
    restarted_store = SemanticStore.from_memory_root(
        restarted_memory.settings.memory.persist_directory
    )
    recovered = MemorySemanticParticipant.from_pending(
        restarted_memory,
        restarted_store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=binding.event_id,
        processing_sequence=binding.processing_sequence,
    )

    assert recovered.finalize(binding) is ParticipantOutcome.FINALIZED
    assert recovered.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert sorted(record_directory.glob("*.json")) == [record_directory / "0.json"]
    assert restarted_store.load_pending(binding.transaction_id) is None


def test_committed_semantic_recovery_does_not_read_live_db1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    source = SemanticSourceEdge(
        SemanticSourceKind.EPISODIC,
        "episode-source",
        captured_context_id="context-source",
        source_status=SemanticSourceStatus.AVAILABLE,
    )
    participant, binding, revision = _create_participant(
        memory, source_edges=(source,)
    )
    committed = SimpleNamespace(record=SimpleNamespace(context_id="context-source"))
    monkeypatch.setattr(
        memory, "get_committed_episodic", lambda _source_id: committed
    )
    participant.prepare(binding)
    participant.store.publish_create(revision, participant.operation_digest)

    def unavailable(_source_id: str) -> None:
        raise EpisodicMemoryReadError("DB1 is unavailable")

    monkeypatch.setattr(memory, "get_committed_episodic", unavailable)
    recovered = MemorySemanticParticipant.from_pending(
        memory,
        participant.store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=binding.event_id,
        processing_sequence=binding.processing_sequence,
    )
    assert recovered.finalize(binding) is ParticipantOutcome.FINALIZED
    assert recovered.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT


def test_exact_semantic_source_revision_is_validated_and_recovery_does_not_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    source = _revision_for_slot("88888888-8888-4888-8888-888888888888", 0, 7)
    participant, binding, revision = _create_participant(
        memory,
        source_edges=(
            SemanticSourceEdge(
                SemanticSourceKind.SEMANTIC,
                source.semantic_id,
                source_revision=0,
                source_status=SemanticSourceStatus.AVAILABLE,
            ),
        ),
    )
    participant.store.publish_create(source, "c" * 64, batch_index=0)

    participant.prepare(binding)
    participant.store.publish_create(revision, participant.operation_digest)

    original_load_revision = participant.store.load_revision

    def unavailable(semantic_id: str, revision_number: int) -> object:
        if semantic_id == source.semantic_id:
            raise RuntimeError("source must not be reread after commit")
        return original_load_revision(semantic_id, revision_number)

    monkeypatch.setattr(participant.store, "load_revision", unavailable)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED


@pytest.mark.parametrize("source_case", ["absent", "malformed", "unavailable"])
def test_semantic_source_revision_failures_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_case: str
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    source = _revision_for_slot("77777777-7777-4777-8777-777777777777", 0, 7)
    edge = SemanticSourceEdge(
        SemanticSourceKind.SEMANTIC,
        source.semantic_id,
        source_revision=1,
        source_status=SemanticSourceStatus.AVAILABLE,
    )
    participant, binding, revision = _create_participant(memory, source_edges=(edge,))
    if source_case == "malformed":
        participant.store.publish_create(source, "d" * 64, batch_index=0)
        participant.store.record_path(source.semantic_id, 1).parent.mkdir(
            parents=True, exist_ok=True
        )
        participant.store.record_path(source.semantic_id, 1).write_text("not json")
    elif source_case == "unavailable":
        monkeypatch.setattr(
            participant.store,
            "load_revision",
            lambda *_args: (_ for _ in ()).throw(SemanticStoreUnavailable("unavailable")),
        )

    expected = ParticipantUnavailableError if source_case != "malformed" else ParticipantDivergedError
    with pytest.raises(expected):
        participant.prepare(binding)
    assert participant.store.load_pending(binding.transaction_id) is None
    assert participant.store.load_current(revision.semantic_id) is None


def test_unpublished_semantic_still_requires_live_source_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    source = SemanticSourceEdge(
        SemanticSourceKind.EPISODIC,
        "episode-source",
        captured_context_id="context-source",
        source_status=SemanticSourceStatus.AVAILABLE,
    )
    participant, binding, _revision = _create_participant(
        memory, source_edges=(source,)
    )

    def unavailable(_source_id: str) -> None:
        raise EpisodicMemoryReadError("DB1 is unavailable")

    monkeypatch.setattr(memory, "get_committed_episodic", unavailable)
    with pytest.raises(ParticipantUnavailableError, match="source is unavailable"):
        participant.prepare(binding)
    assert participant.store.load_pending(binding.transaction_id) is None


def test_divergent_legacy_projection_is_not_overwritten(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    participant.prepare(binding)
    participant.store.publish_create(revision, participant.operation_digest)
    # Chroma IDs are opaque; use the authoritative deterministic identity for
    # the collision row and retain the exact legacy bytes for the assertion.
    collision_id = revision.semantic_id
    memory.db2.add(
        ids=[collision_id],
        documents=["legacy collision"],
        metadatas=[
            {
                "text": "legacy collision",
                "source_episode_ids": "[]",
                "record_type": "semantic_memory",
                "created_at": "legacy",
                "extra": "{}",
            }
        ],
    )
    before = memory.db2.get(ids=[collision_id], include=["documents", "metadatas"])

    with pytest.raises(ParticipantDivergedError):
        participant.finalize(binding)

    assert memory.db2.get(ids=[collision_id], include=["documents", "metadatas"]) == before


@pytest.mark.parametrize(
    ("lifecycle", "operation", "reason"),
    [
        (
            SemanticLifecycle.RETRACTED,
            SemanticRevisionOperation.RETRACT,
            SemanticRevisionReason.RETRACTION,
        ),
        (
            SemanticLifecycle.SUPERSEDED,
            SemanticRevisionOperation.SUPERSEDE,
            SemanticRevisionReason.SUPERSESSION,
        ),
        (
            SemanticLifecycle.ARCHIVED,
            SemanticRevisionOperation.ARCHIVE,
            SemanticRevisionReason.ARCHIVAL,
        ),
        (
            SemanticLifecycle.QUARANTINED,
            SemanticRevisionOperation.QUARANTINE,
            SemanticRevisionReason.QUARANTINE,
        ),
    ],
)
def test_non_active_revision_without_db2_row_is_exact_and_reconciliation_is_noop(
    tmp_path: Path,
    lifecycle: SemanticLifecycle,
    operation: SemanticRevisionOperation,
    reason: SemanticRevisionReason,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    create, create_binding, created = _create_participant(memory)
    create.prepare(create_binding)
    create.finalize(create_binding)
    event = _event(2)
    transaction_id = TransactionCoordinator.derive_transaction_id(
        event, TransactionKind.EVENT_MUTATION
    )
    revised = SemanticRevision(
        semantic_id=created.semantic_id,
        revision=1,
        semantic_content=created.semantic_content,
        content_digest=created.content_digest,
        created_at=event.requested_at,
        lifecycle=lifecycle,
        operation=operation,
        reason=reason,
        previous_revision_digest=created.revision_digest,
        event_id=event.event_id,
        event_sequence=event.processing_sequence,
    )
    operation = SemanticBatchOperation(
        transaction_id,
        (SemanticBatchEntry(0, SemanticRevisionIntent(revised, 0, created.revision_digest)),),
    )
    participant = MemorySemanticParticipant(memory, create.store, operation)
    binding = TransactionBinding(
        transaction_id,
        event.event_id,
        event.processing_sequence or 0,
        participant.participant_id,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    participant.prepare(binding)
    participant.finalize(binding)

    assert memory.db2.get(ids=[created.semantic_id], include=["documents", "metadatas"])["ids"] == []
    assert (
        participant.inspect_reconciliation(binding)
        is StartupParticipantOutcome.VERIFIED_CONSISTENT
    )
    assert (
        participant.reconcile(binding) is StartupParticipantOutcome.VERIFIED_CONSISTENT
    )
    assert memory.db2.get(ids=[created.semantic_id], include=["documents", "metadatas"])["ids"] == []


def test_oversized_batch_rejected_before_publication() -> None:
    transaction_id = "33333333-3333-4333-8333-333333333333"
    with pytest.raises(ValueError):
        SemanticBatchOperation(
            transaction_id,
            tuple(
                SemanticBatchEntry(
                    index,
                    SemanticCreateIntent(_revision_for_slot(transaction_id, index, 1)),
                )
                for index in range(129)
            ),
        )
