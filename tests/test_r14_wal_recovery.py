"""U4 Attention continuity through StateWAL and startup recovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from suzka.attention.contracts import (
    AttentionCandidateProjection,
    AttentionEvent,
)
from suzka.attention.system import AttentionSystem
from suzka.runtime.agent_runtime import AgentEvent
from suzka.runtime.agent_state import (
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshotV5,
    AgentStateSnapshotV7,
    AgentStateSnapshotV8,
    AgentStateSnapshotV9,
    CompatibleAgentStateSnapshot,
    AgentStateStore,
)
from suzka.runtime.attention_state_port import AttentionStatePort
from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalIntegrityError,
    EventJournalTransaction,
    EventLifecycle,
)
from suzka.runtime.state_recovery import (
    InternalCommitClassification,
    InternalCommitEvidence,
    StateRecoveryCoordinator,
    StateRecoveryError,
)
from suzka.runtime.state_wal import (
    BaselineRecord,
    StateWAL,
    StateWALConflictError,
    StateWALError,
    StateWALIntegrityError,
    TransitionRecord,
)
from test_agent_state_context import as_v3, as_v4, make_context_loop
from test_attention_system import _goal_projection
from test_r13_wal_recovery import (
    NOW,
    _Loop,
    _R13Loop,
    _boot_from_snapshot,
    _capture_v8,
    _corrupt_active_transition,
    _domains,
    _event,
    _recovery_components,
    _start_event,
    _store,
)
from test_state_recovery import start_transaction as _start_classifier_transaction
from test_state_wal import make_snapshot, make_v1_snapshot, make_v6_snapshot


def _attention_event(sequence: int) -> AttentionEvent:
    return AttentionEvent(
        f"u4-attention-event:{sequence}",
        sequence,
        NOW,
    )


def _seed_attention(
    through_sequence: int = 18,
) -> tuple[
    AttentionSystem,
    tuple[AttentionEvent, ...],
    tuple[AttentionCandidateProjection, ...],
]:
    system = AttentionSystem()
    events: list[AttentionEvent] = []
    projections: list[AttentionCandidateProjection] = []
    for sequence in range(1, through_sequence + 1):
        event = _attention_event(sequence)
        projection = _goal_projection(event)
        system.refresh((projection,), event)
        events.append(event)
        projections.append(projection)
    return system, tuple(events), tuple(projections)


def _attach_ports(
    owner: _R13Loop,
    attention: AttentionSystem,
) -> _R13Loop:
    owner.agent_state_ports = SimpleNamespace(
        motivation_state_port=owner,
        goal_state_port=owner,
        commitment_state_port=owner,
        attention_state_port=AttentionStatePort(attention),
    )
    return owner


def _capture_v9(
    store: AgentStateStore,
    sequence: int,
    attention: AttentionSystem,
    *,
    populated_r13: bool = True,
) -> tuple[AgentStateSnapshotV9, _R13Loop]:
    owner = _attach_ports(
        _R13Loop(_domains(populated=populated_r13)),
        attention,
    )
    snapshot = store.capture(owner, sequence=sequence)
    assert isinstance(snapshot, AgentStateSnapshotV9)
    return snapshot, owner


def _initial_v9(
    store: AgentStateStore,
    sequence: int = 20,
) -> tuple[
    AgentStateSnapshotV9,
    _R13Loop,
    AttentionSystem,
    tuple[AttentionEvent, ...],
    tuple[AttentionCandidateProjection, ...],
]:
    attention, events, projections = _seed_attention()
    snapshot, owner = _capture_v9(store, sequence, attention)
    assert snapshot.attention_state == attention.snapshot()
    return snapshot, owner, attention, events, projections


def _next_v9(
    store: AgentStateStore,
    prior: AgentStateSnapshotV9,
    sequence: int,
) -> tuple[
    AgentStateSnapshotV9,
    _R13Loop,
    AttentionSystem,
    AttentionEvent,
    AttentionCandidateProjection,
]:
    attention = AttentionSystem(prior.attention_state)
    event = _attention_event(sequence)
    projection = _goal_projection(event)
    attention.refresh((projection,), event)
    snapshot, owner = _capture_v9(store, sequence, attention)
    return snapshot, owner, attention, event, projection


def _commit(
    recovery: StateRecoveryCoordinator,
    journal: EventJournal,
    event: AgentEvent,
    prior: CompatibleAgentStateSnapshot,
    candidate: CompatibleAgentStateSnapshot,
    *,
    complete: bool = True,
) -> InternalCommitEvidence:
    _start_event(journal, event)
    evidence = recovery.commit_internal_candidate(event, prior, candidate)
    if complete:
        recovery.complete_committed_event(event, evidence)
    return evidence


def _prepared_event_lifecycle(
    journal: EventJournal,
    event: AgentEvent,
) -> EventLifecycle:
    inspection = journal.inspect()
    prepared = next(
        record
        for record in reversed(inspection.records)
        if record.lifecycle is EventLifecycle.PREPARED
        and record.event_id == event.event_id
        and record.processing_sequence == event.processing_sequence
    )
    assert prepared.lifecycle is EventLifecycle.PREPARED
    assert prepared.event_type is event.event_type
    assert prepared.source is event.source
    open_event = next(
        item for item in inspection.open_events if item.event_id == event.event_id
    )
    assert open_event.lifecycle is EventLifecycle.PREPARED
    assert open_event.processing_sequence == event.processing_sequence
    return prepared.lifecycle


def _event_classifier_transaction(
    journal: EventJournal,
    event: AgentEvent,
) -> EventJournalTransaction | None:
    inspection = journal.inspect()
    return next(
        (
            transaction
            for transaction in inspection.open_transactions
            if transaction.event_id == event.event_id
            and transaction.processing_sequence == event.processing_sequence
            and transaction.event_type is event.event_type
            and transaction.source is event.source
        ),
        None,
    )


def _legacy_candidate(
    store: AgentStateStore,
    schema_version: int,
    sequence: int = 21,
) -> CompatibleAgentStateSnapshot:
    if schema_version == 1:
        return make_v1_snapshot(sequence)
    if schema_version == 2:
        return make_snapshot(sequence)

    context_snapshot = store.capture(make_context_loop(), sequence=sequence)
    assert isinstance(context_snapshot, AgentStateSnapshotV5)
    if schema_version == 3:
        return as_v3(context_snapshot)
    if schema_version == 4:
        return as_v4(context_snapshot)
    if schema_version == 5:
        return context_snapshot
    if schema_version == 6:
        return make_v6_snapshot(sequence)
    if schema_version == 7:
        snapshot = store.capture(_Loop(), sequence=sequence)
        assert isinstance(snapshot, AgentStateSnapshotV7)
        return snapshot
    if schema_version == 8:
        snapshot, _source = _capture_v8(store, populated=True)
        return AgentStateSnapshotV8.model_validate(
            {
                **snapshot.model_dump(mode="python"),
                "last_processed_event_sequence": sequence,
            }
        )
    raise AssertionError(f"unexpected legacy schema version {schema_version}")


def _boot_v9(
    tmp_path: Path,
    store: AgentStateStore,
    snapshot: AgentStateSnapshotV9,
) -> tuple[StateRecoveryCoordinator, EventJournal, StateWAL]:
    store.save(snapshot)
    recovery, journal, wal = _recovery_components(tmp_path, store)
    snapshot_hash = store.snapshot_hash(snapshot)
    journal.verify_and_reconcile(
        snapshot.last_processed_event_sequence,
        snapshot_hash,
    )
    wal.bootstrap(snapshot, snapshot.last_processed_event_sequence)
    bootable = recovery.prepare_startup()
    assert bootable.snapshot == snapshot
    recovery.publish_boot_anchor(bootable)
    return recovery, journal, wal


def test_mixed_v8_v9_v9_wal_reconstructs_exact_attention_roots_and_receipts(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    v8, _source = _capture_v8(store, populated=True)
    v8 = AgentStateSnapshotV8.model_validate(
        {
            **v8.model_dump(mode="python"),
            "last_processed_event_sequence": 19,
        }
    )

    attention, events, projections = _seed_attention()
    replay = attention.refresh((projections[-1],), events[-1])
    assert replay.replayed is True
    assert replay.snapshot == attention.snapshot()
    first_event = _attention_event(20)
    first_projection = _goal_projection(first_event)
    attention.refresh((first_projection,), first_event)
    first, _first_owner = _capture_v9(store, 20, attention)

    second_attention = AttentionSystem(first.attention_state)
    second_event = _attention_event(21)
    second_projection = _goal_projection(second_event)
    second_attention.refresh((second_projection,), second_event)
    second, _second_owner = _capture_v9(store, 21, second_attention)

    wal = StateWAL(tmp_path / "wal")
    manifest = wal.bootstrap(v8, 19)
    first_transition = wal.append_transition(
        event_id=uuid5(NAMESPACE_URL, "r14-mixed-v8-v9-first"),
        event_type="state.transition",
        event_source="test",
        processing_sequence=20,
        prior_snapshot=v8,
        candidate_snapshot=first,
    )
    second_transition = wal.append_transition(
        event_id=uuid5(NAMESPACE_URL, "r14-mixed-v9-second"),
        event_type="state.transition",
        event_source="test",
        processing_sequence=21,
        prior_snapshot=first,
        candidate_snapshot=second,
    )

    inspection = wal.inspect()
    assert len(inspection.records) == 3
    baseline_record, first_record, second_record = inspection.records
    assert isinstance(baseline_record, BaselineRecord)
    assert isinstance(first_record, TransitionRecord)
    assert isinstance(second_record, TransitionRecord)
    assert baseline_record.baseline_snapshot == v8
    assert first_record.candidate_snapshot == first
    assert second_record.candidate_snapshot == second
    assert first_transition.previous_record_hash == manifest.active_baseline_record_hash
    assert second_transition.previous_record_hash == first_transition.record_hash
    assert inspection.record_hashes == (
        manifest.active_baseline_record_hash,
        first_transition.record_hash,
        second_transition.record_hash,
    )

    for snapshot, record in (
        (v8, baseline_record),
        (first, first_record),
        (second, second_record),
    ):
        snapshot_hash = store.snapshot_hash(snapshot)
        assert wal.reconstruct(
            sequence=snapshot.last_processed_event_sequence,
            snapshot_hash=snapshot_hash,
            record_id=record.record_id,
        ) == snapshot
        reconstructed = wal.reconstruct(record_id=record.record_id)
        assert store.snapshot_hash(reconstructed) == snapshot_hash

    attention_root = second.attention_state
    assert attention_root.revision == 20
    assert attention_root.revision_anchor is not None
    assert attention_root.receipts == second_attention.snapshot().receipts
    assert attention_root.candidates[0].focused_event_count > 0
    assert attention_root.focused_ids
    assert attention_root.unfinished_ids == ()
    assert attention_root.receipts[-1].event == second_event
    assert second.attention_state.canonical_bytes() == (
        second_attention.snapshot().canonical_bytes()
    )


def test_first_normal_v8_to_v9_commit_is_lazy_then_publishes_canonical_attention(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, _source = _capture_v8(store, populated=True)
    store.save(prior)
    retained_bytes = store.path.read_bytes()
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, prior)
    assert store.path.read_bytes() == retained_bytes

    attention = AttentionSystem()
    event = _attention_event(21)
    projection = _goal_projection(event)
    attention.refresh((projection,), event)
    candidate, _owner = _capture_v9(store, 21, attention)
    assert isinstance(candidate, AgentStateSnapshotV9)

    agent_event = _event("first-normal-v9-after-v8", 21)
    evidence = _commit(recovery, journal, agent_event, prior, candidate)

    inspection = wal.inspect()
    assert len(inspection.records) == 2
    baseline_record, transition = inspection.records
    assert isinstance(baseline_record, BaselineRecord)
    assert isinstance(transition, TransitionRecord)
    assert baseline_record.baseline_snapshot == prior
    assert transition.candidate_snapshot == candidate
    assert transition.prior_snapshot_hash == store.snapshot_hash(prior)
    assert transition.candidate_snapshot_hash == evidence.snapshot_hash
    assert store.load() == candidate
    assert isinstance(store.load(), AgentStateSnapshotV9)
    assert store.path.read_bytes() != retained_bytes
    assert store.path.read_bytes() == store.canonical_bytes(candidate)


@pytest.mark.parametrize("schema_version", range(1, 9))
def test_ordinary_v9_to_legacy_transition_is_rejected_by_wal_without_mutation(
    tmp_path: Path,
    schema_version: int,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, _owner, _attention, _events, _projections = _initial_v9(store)
    candidate = _legacy_candidate(store, schema_version)
    assert candidate.schema_version == schema_version
    store.save(prior)

    wal = StateWAL(tmp_path / "wal")
    manifest = wal.bootstrap(prior, prior.last_processed_event_sequence)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    before_inspection = wal.inspect()
    before_generation = generation.read_bytes()
    before_canonical = store.path.read_bytes()

    with pytest.raises(StateWALConflictError, match="downgrade schema"):
        wal.append_transition(
            event_id=uuid5(NAMESPACE_URL, f"ordinary-v9-to-v{schema_version}"),
            event_type="state.transition",
            event_source="test",
            processing_sequence=candidate.last_processed_event_sequence,
            prior_snapshot=prior,
            candidate_snapshot=candidate,
        )

    assert wal.inspect() == before_inspection
    assert generation.read_bytes() == before_generation
    assert store.path.read_bytes() == before_canonical
    assert store.load() == prior


@pytest.mark.parametrize("schema_version", range(1, 9))
def test_ordinary_v9_internal_commit_rejects_legacy_before_journal_prepare(
    tmp_path: Path,
    schema_version: int,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, _owner, _attention, _events, _projections = _initial_v9(store)
    candidate = _legacy_candidate(store, schema_version)
    recovery, journal, wal = _boot_v9(tmp_path, store, prior)
    event = _event(f"ordinary-v9-to-v{schema_version}-internal", 21)
    _start_event(journal, event)

    before_journal = journal.path.read_bytes()
    before_inspection = wal.inspect()
    manifest = before_inspection.active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    before_generation = generation.read_bytes()
    before_canonical = store.path.read_bytes()

    with pytest.raises(StateRecoveryError, match="downgrade schema"):
        recovery.commit_internal_candidate(event, prior, candidate)

    assert journal.path.read_bytes() == before_journal
    assert wal.inspect() == before_inspection
    assert generation.read_bytes() == before_generation
    assert store.path.read_bytes() == before_canonical
    assert store.load() == prior

    # A rejected candidate leaves the retained current generation repeatedly
    # recoverable; it never turns the refusal into an implicit rollback.
    for _ in range(2):
        recovered = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
        assert recovered.snapshot == prior
        assert wal.reconstruct() == prior


def test_true_rollback_from_v9_restores_exact_attention_history_and_r13_ports(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, prior_owner, _attention, events, projections = _initial_v9(store)
    recovery, journal, wal = _boot_v9(tmp_path / "recovery", store, prior)
    candidate, _candidate_owner, _new_attention, _event_value, _projection = _next_v9(
        store, prior, 21
    )
    _commit(recovery, journal, _event("v9-true-rollback-newer", 21), prior, candidate)
    _corrupt_active_transition(wal)
    store.path.unlink()

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert rolled_back.snapshot == prior
    assert rolled_back.snapshot_hash == store.snapshot_hash(prior)
    assert rolled_back.true_rollback_performed is True
    assert rolled_back.external_reconciliation_required is True
    retained_baseline = wal.inspect().records[0]
    assert isinstance(retained_baseline, BaselineRecord)
    assert retained_baseline.baseline_snapshot == prior
    active_manifest = wal.inspect().active_manifest
    assert active_manifest is not None
    assert active_manifest.external_reconciliation_required is True
    assert journal.inspect().external_reconciliation_required is True

    target_attention = AttentionSystem(candidate.attention_state)
    target_owner = _attach_ports(
        _R13Loop(_domains(populated=False)),
        target_attention,
    )
    store.restore_into(target_owner, rolled_back.snapshot)
    assert target_attention.snapshot().canonical_bytes() == (
        prior.attention_state.canonical_bytes()
    )
    assert target_owner.r13_state() == prior_owner.r13_state()
    restored_view = target_attention.selected_view()
    assert restored_view.focused_targets
    assert restored_view.competition is None
    assert restored_view.prompt is None
    replay = target_attention.refresh((projections[-1],), events[-1])
    assert replay.replayed is True
    assert replay.snapshot.canonical_bytes() == (
        prior.attention_state.canonical_bytes()
    )


def test_true_rollback_from_v9_to_retained_v8_bootstraps_attention_and_keeps_r13(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, prior_source = _capture_v8(store, populated=True)
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, prior)
    attention = AttentionSystem()
    attention_event = _attention_event(21)
    attention.refresh((_goal_projection(attention_event),), attention_event)
    candidate, _candidate_owner = _capture_v9(store, 21, attention)
    _commit(recovery, journal, _event("v9-true-rollback-to-v8", 21), prior, candidate)
    _corrupt_active_transition(wal)
    store.path.unlink()

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert rolled_back.snapshot == prior
    assert rolled_back.true_rollback_performed is True
    assert rolled_back.external_reconciliation_required is True

    target_attention = AttentionSystem(candidate.attention_state)
    target_owner = _attach_ports(
        _R13Loop(_domains(populated=False)),
        target_attention,
    )
    store.restore_into(target_owner, rolled_back.snapshot)
    restored = target_attention.snapshot()
    assert restored.revision == 0
    assert restored.last_event is None
    assert restored.revision_history == ()
    assert restored.receipts == ()
    assert restored.focused_ids == ()
    assert restored.unfinished_ids == ()
    view = target_attention.selected_view()
    assert view.event is None
    assert view.focused_targets == ()
    assert view.competition is None
    assert view.prompt is None
    assert target_owner.r13_state() == prior_source.r13_state()


@pytest.mark.parametrize("failure_boundary", ["before_wal", "before_canonical"])
@pytest.mark.parametrize(
    "external_transaction",
    [False, True],
    ids=["internal-event-only", "external-transaction-open"],
)
def test_v9_crash_before_internal_commit_recovers_prior_v8_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
    external_transaction: bool,
) -> None:
    path = tmp_path / "agent_state.json"
    if failure_boundary == "before_canonical":
        fail_publication = False

        def fail_before_replace(stage: AgentStateSaveStage) -> None:
            if fail_publication and stage is AgentStateSaveStage.ATOMIC_REPLACE:
                raise RuntimeError("injected v9 publication failure")

        store = _store(path, hook=fail_before_replace)
    else:
        store = _store(path)

    prior, _source = _capture_v8(store, populated=True)
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, prior)
    attention = AttentionSystem()
    attention_event = _attention_event(21)
    attention.refresh((_goal_projection(attention_event),), attention_event)
    candidate, _owner = _capture_v9(store, 21, attention)
    event = _event(f"crash-{failure_boundary}-v9", 21)
    if external_transaction:
        _start_classifier_transaction(journal, event)
    else:
        _start_event(journal, event)

    if failure_boundary == "before_wal":
        def fail_wal_append(**_kwargs: object) -> object:
            raise StateWALError("injected failure before WAL append")

        monkeypatch.setattr(wal, "append_transition", fail_wal_append)
        with pytest.raises(StateWALError, match="injected failure"):
            recovery.commit_internal_candidate(event, prior, candidate)
    else:
        fail_publication = True
        with pytest.raises(AgentStateSaveError) as error:
            recovery.commit_internal_candidate(event, prior, candidate)
        assert error.value.published is False

    assert _prepared_event_lifecycle(journal, event) is EventLifecycle.PREPARED
    transaction = _event_classifier_transaction(journal, event)
    if external_transaction:
        assert transaction is not None
        proof = recovery.classify_transaction_commit(transaction)
        assert proof.classification is InternalCommitClassification.PRE_INTERNAL
    else:
        assert transaction is None
    assert store.load() == prior

    if external_transaction:
        journal_before = journal.path.read_bytes()
        wal_before = wal.inspect()
        canonical_before = store.path.read_bytes()
        with pytest.raises(EventJournalIntegrityError):
            StateRecoveryCoordinator(_store(path), journal, wal).prepare_startup()
        assert journal.path.read_bytes() == journal_before
        assert wal.inspect() == wal_before
        assert store.path.read_bytes() == canonical_before
        assert transaction is not None
        assert transaction in journal.inspect().open_transactions
    else:
        restarted = StateRecoveryCoordinator(_store(path), journal, wal).prepare_startup()
        assert restarted.snapshot == prior
        assert restarted.snapshot_hash == store.snapshot_hash(prior)
        assert _store(path).load() == prior
        assert wal.inspect().latest_snapshot_hash == store.snapshot_hash(prior)


@pytest.mark.parametrize(
    "external_transaction",
    [False, True],
    ids=["internal-event-only", "external-transaction-open"],
)
def test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_transaction: bool,
) -> None:
    import suzka.attention.system as attention_system_module
    import suzka.metacognition.assessment as metacognition_assessment

    store = _store(tmp_path / "agent_state.json")
    prior, _source = _capture_v8(store, populated=True)
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, prior)
    attention = AttentionSystem()
    attention_event = _attention_event(21)
    attention.refresh((_goal_projection(attention_event),), attention_event)
    candidate, _owner = _capture_v9(store, 21, attention)
    event = _event("canonical-v9-before-journal-completion", 21)
    if external_transaction:
        _start_classifier_transaction(journal, event)
    else:
        _start_event(journal, event)
    evidence = recovery.commit_internal_candidate(event, prior, candidate)
    assert store.load() == candidate
    assert _prepared_event_lifecycle(journal, event) is EventLifecycle.PREPARED
    transaction = _event_classifier_transaction(journal, event)
    if external_transaction:
        assert transaction is not None
        assert recovery.classify_transaction_commit(transaction).classification is (
            InternalCommitClassification.INTERNALLY_COMMITTED
        )
    else:
        assert transaction is None

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("U4 startup recovery replayed a policy or assessment operation")

    monkeypatch.setattr(AttentionSystem, "refresh", forbidden)
    monkeypatch.setattr(attention_system_module, "compete_attention", forbidden)
    monkeypatch.setattr(attention_system_module, "select_attention_prompt", forbidden)
    monkeypatch.setattr(metacognition_assessment, "assess_metacognition", forbidden)

    if external_transaction:
        journal_before = journal.path.read_bytes()
        wal_before = wal.inspect()
        canonical_before = store.path.read_bytes()
        with pytest.raises(EventJournalIntegrityError):
            StateRecoveryCoordinator(store, journal, wal).prepare_startup()
        assert journal.path.read_bytes() == journal_before
        assert wal.inspect() == wal_before
        assert store.path.read_bytes() == canonical_before
        assert transaction is not None
        assert transaction in journal.inspect().open_transactions
    else:
        restarted = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
        assert restarted.snapshot == candidate
        assert restarted.snapshot_hash == evidence.snapshot_hash
        assert restarted.processing_high_water == 21
        assert wal.reconstruct(
            sequence=21,
            snapshot_hash=evidence.snapshot_hash,
        ) == candidate


def test_journal_bound_prefix_repairs_deleted_v9_canonical_without_replaying_attention(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, _source = _capture_v8(store, populated=True)
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, prior)
    attention = AttentionSystem()
    attention_event = _attention_event(21)
    attention.refresh((_goal_projection(attention_event),), attention_event)
    candidate, _owner = _capture_v9(store, 21, attention)
    _commit(recovery, journal, _event("journal-bound-v9-repair", 21), prior, candidate)

    before = wal.inspect()
    manifest = before.active_manifest
    assert manifest is not None
    transition = before.records[-1]
    assert isinstance(transition, TransitionRecord)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    retained_generation = generation.read_bytes()
    generation.write_bytes(retained_generation + b'{"truncated":"tail"}\n')
    damaged_generation = generation.read_bytes()
    generation.chmod(0o600)
    assert wal.inspect_bound_prefix(
        generation_id=manifest.active_generation_id,
        record_id=transition.record_id,
        record_hash=transition.record_hash,
        snapshot_sequence=21,
        snapshot_hash=store.snapshot_hash(candidate),
    ) == candidate
    store.path.unlink()

    repaired = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert repaired.snapshot == candidate
    assert repaired.snapshot_hash == store.snapshot_hash(candidate)
    assert repaired.exact_current_reconstructed is True
    assert store.load() == candidate
    assert wal.inspect().latest_snapshot_hash == store.snapshot_hash(candidate)
    assert generation.read_bytes() == damaged_generation


@pytest.mark.parametrize("tamper", ["duplicate_schema", "future_schema"])
def test_raw_v9_attention_tampering_is_rejected_before_wal_repair_or_truncation(
    tmp_path: Path,
    tamper: str,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _owner, _attention, _events, _projections = _initial_v9(store)
    wal = StateWAL(tmp_path / "wal")
    manifest = wal.bootstrap(snapshot, snapshot.last_processed_event_sequence)
    baseline = wal.inspect().records[0]
    assert isinstance(baseline, BaselineRecord)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    original = generation.read_bytes()

    attention_marker = b'"attention_state":{'
    marker_index = original.index(attention_marker) + len(attention_marker)
    tampered = original
    if tamper == "duplicate_schema":
        tampered = (
            original[:marker_index]
            + b'"schema_version":1,'
            + original[marker_index:]
        )
    else:
        nested_schema_index = original.index(b'"schema_version":1', marker_index)
        tampered = (
            original[:nested_schema_index]
            + b'"schema_version":999'
            + original[nested_schema_index + len(b'"schema_version":1') :]
        )
    generation.write_bytes(tampered)
    generation.chmod(0o600)

    with pytest.raises(StateWALIntegrityError):
        wal.inspect()
    with pytest.raises(StateWALIntegrityError):
        wal.inspect_bound_prefix(
            generation_id=manifest.active_generation_id,
            record_id=baseline.record_id,
            record_hash=baseline.record_hash,
            snapshot_sequence=snapshot.last_processed_event_sequence,
            snapshot_hash=store.snapshot_hash(snapshot),
        )
    assert generation.read_bytes() == tampered


def test_deep_non_attention_wal_json_is_translated_to_bounded_wal_errors(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _owner, _attention, _events, _projections = _initial_v9(store)
    wal = StateWAL(tmp_path / "wal")
    manifest = wal.bootstrap(snapshot, snapshot.last_processed_event_sequence)
    before = wal.inspect()
    baseline = before.records[0]
    assert isinstance(baseline, BaselineRecord)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    original = generation.read_bytes()
    deep_non_attention_value = b"[" * 1_500 + b"0" + b"]" * 1_500
    tampered = (
        original[:-2]
        + b',"non_attention_extension":'
        + deep_non_attention_value
        + b"}\n"
    )
    generation.write_bytes(tampered)
    generation.chmod(0o600)

    with pytest.raises(StateWALIntegrityError):
        wal.inspect()
    with pytest.raises(StateWALIntegrityError):
        wal.inspect_bound_prefix(
            generation_id=manifest.active_generation_id,
            record_id=baseline.record_id,
            record_hash=baseline.record_hash,
            snapshot_sequence=snapshot.last_processed_event_sequence,
            snapshot_hash=store.snapshot_hash(snapshot),
        )
    with pytest.raises(StateWALConflictError):
        wal._inspect_orphan_baseline(manifest.active_generation_id)
    assert generation.read_bytes() == tampered


def test_mutated_typed_attention_checksum_fails_wal_save_and_commit_before_prepare(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, _owner, _attention, _events, _projections = _initial_v9(store)
    candidate, _candidate_owner, _new_attention, _event_value, _projection = _next_v9(
        store, prior, 21
    )
    store.save(prior)
    before_canonical = store.path.read_bytes()
    wal = StateWAL(tmp_path / "wal")
    manifest = wal.bootstrap(prior, prior.last_processed_event_sequence)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    before_generation = generation.read_bytes()

    object.__setattr__(
        candidate.attention_state.revision_history[-1],
        "state_digest",
        "0" * 64,
    )
    with pytest.raises(StateWALError):
        wal.append_transition(
            event_id=uuid5(NAMESPACE_URL, "typed-attention-tamper-wal"),
            event_type="state.transition",
            event_source="test",
            processing_sequence=21,
            prior_snapshot=prior,
            candidate_snapshot=candidate,
        )
    with pytest.raises(AgentStateSaveError):
        store.save(candidate)

    recovery, journal, wal = _boot_v9(tmp_path / "recovery", store, prior)
    event = _event("typed-attention-tamper-before-prepare", 21)
    _start_event(journal, event)
    before_journal = journal.path.read_bytes()
    before_inspection = wal.inspect()

    with pytest.raises((AgentStateSaveError, StateRecoveryError)):
        recovery.commit_internal_candidate(event, prior, candidate)

    assert journal.path.read_bytes() == before_journal
    assert wal.inspect() == before_inspection
    assert generation.read_bytes() == before_generation
    assert store.path.read_bytes() == before_canonical
    assert store.load() == prior
