"""U5 R13 snapshot continuity through StateWAL and startup recovery."""

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from suzka.motivation.commitment_system import CommitmentSystem
from suzka.motivation.goal_system import GoalSystem
from suzka.motivation.system import MotivationSystem
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from suzka.runtime.agent_state import (
    AgentStateLoadError,
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshotV7,
    AgentStateSnapshotV8,
    AgentStateStore,
)
from suzka.runtime.event_journal import EventJournal
from suzka.runtime.state_recovery import StateRecoveryCoordinator
from suzka.runtime.state_wal import StateWAL, StateWALError
from test_agent_state_v8 import (
    NOW,
    _Loop,
    _R13Loop,
    _capture_v8,
    _domains,
    _store,
)


def _event(name: str, sequence: int) -> AgentEvent:
    return AgentEvent(
        str(uuid5(NAMESPACE_URL, name)),
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        NOW,
        sequence,
    )


def _start_event(journal: EventJournal, event: AgentEvent) -> None:
    journal.append_accepted(event)
    journal.append_started(event)


def _recovery_components(
    tmp_path: Path, store: AgentStateStore
) -> tuple[StateRecoveryCoordinator, EventJournal, StateWAL]:
    journal = EventJournal(
        tmp_path / "events.jsonl", 100_000, 4, clock=lambda: NOW
    )
    wal = StateWAL(tmp_path / "wal")
    return StateRecoveryCoordinator(store, journal, wal), journal, wal


def _boot_from_snapshot(
    tmp_path: Path,
    store: AgentStateStore,
    initial: AgentStateSnapshotV7 | AgentStateSnapshotV8,
) -> tuple[StateRecoveryCoordinator, EventJournal, StateWAL]:
    store.save(initial)
    recovery, journal, wal = _recovery_components(tmp_path, store)
    initial_hash = store.snapshot_hash(initial)
    journal.verify_and_reconcile(initial.last_processed_event_sequence, initial_hash)
    wal.bootstrap(initial, initial.last_processed_event_sequence)

    bootable = recovery.prepare_startup()
    assert bootable.snapshot == initial
    recovery.publish_boot_anchor(bootable)
    return recovery, journal, wal


def _commit(
    recovery: StateRecoveryCoordinator,
    journal: EventJournal,
    event: AgentEvent,
    prior: AgentStateSnapshotV7 | AgentStateSnapshotV8,
    candidate: AgentStateSnapshotV8,
) -> None:
    _start_event(journal, event)
    evidence = recovery.commit_internal_candidate(event, prior, candidate)
    recovery.complete_committed_event(event, evidence)


def _corrupt_active_transition(wal: StateWAL) -> bytes:
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    generation.chmod(0o600)
    return generation.read_bytes()


def test_mixed_retained_v7_then_v8_wal_reconstructs_exact_snapshots_and_hash_chain(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    retained_v7 = store.capture(_Loop(), sequence=0)
    current_v8, _source = _capture_v8(store, populated=True)
    assert isinstance(retained_v7, AgentStateSnapshotV7)

    wal = StateWAL(tmp_path / "wal")
    baseline = wal.bootstrap(retained_v7, 0)
    transition = wal.append_transition(
        event_id=uuid5(NAMESPACE_URL, "mixed-v7-v8-transition"),
        event_type="state.transition",
        event_source="test",
        processing_sequence=20,
        prior_snapshot=retained_v7,
        candidate_snapshot=current_v8,
    )

    inspection = wal.inspect()
    assert len(inspection.records) == 2
    assert inspection.records[0].baseline_snapshot == retained_v7
    assert inspection.records[0].baseline_snapshot_hash == store.snapshot_hash(
        retained_v7
    )
    assert inspection.records[1].candidate_snapshot == current_v8
    assert inspection.records[1].prior_snapshot_hash == store.snapshot_hash(
        retained_v7
    )
    assert inspection.records[1].candidate_snapshot_hash == store.snapshot_hash(
        current_v8
    )
    assert transition.previous_record_hash == baseline.active_baseline_record_hash
    assert inspection.record_hashes == (
        baseline.active_baseline_record_hash,
        transition.record_hash,
    )
    assert wal.reconstruct(sequence=0) == retained_v7
    assert wal.reconstruct(sequence=20) == current_v8
    assert (
        wal.reconstruct(snapshot_hash=transition.candidate_snapshot_hash)
        == current_v8
    )


def test_first_normal_v8_commit_after_v7_is_lazy_and_preserves_v7_lineage(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    retained_v7 = store.capture(_Loop(), sequence=19)
    assert isinstance(retained_v7, AgentStateSnapshotV7)
    store.save(retained_v7)
    retained_bytes = store.path.read_bytes()

    recovery, journal, wal = _recovery_components(tmp_path, store)
    retained_hash = store.snapshot_hash(retained_v7)
    journal.verify_and_reconcile(19, retained_hash)
    wal.bootstrap(retained_v7, 19)
    bootable = recovery.prepare_startup()

    assert bootable.snapshot == retained_v7
    assert store.path.read_bytes() == retained_bytes
    recovery.publish_boot_anchor(bootable)

    candidate, _source = _capture_v8(store, populated=True)
    event = _event("first-normal-v8-after-v7", 20)
    _start_event(journal, event)
    evidence = recovery.commit_internal_candidate(event, retained_v7, candidate)
    recovery.complete_committed_event(event, evidence)

    inspection = wal.inspect()
    assert len(inspection.records) == 2
    assert inspection.records[0].baseline_snapshot == retained_v7
    assert inspection.records[1].prior_snapshot_hash == retained_hash
    assert inspection.records[1].candidate_snapshot == candidate
    assert isinstance(store.load(), AgentStateSnapshotV8)
    assert store.load() == candidate
    assert store.path.read_bytes() != retained_bytes


def test_failed_first_v8_publication_keeps_canonical_v7_bytes(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    initial_store = _store(path)
    retained_v7 = initial_store.capture(_Loop(), sequence=19)
    assert isinstance(retained_v7, AgentStateSnapshotV7)
    initial_store.save(retained_v7)
    retained_bytes = path.read_bytes()

    def fail_before_replace(stage: AgentStateSaveStage) -> None:
        if stage is AgentStateSaveStage.ATOMIC_REPLACE:
            raise RuntimeError("injected first v8 publication failure")

    failing_store = _store(path, hook=fail_before_replace)
    recovery, journal, wal = _recovery_components(tmp_path, failing_store)
    retained_hash = failing_store.snapshot_hash(retained_v7)
    journal.verify_and_reconcile(19, retained_hash)
    wal.bootstrap(retained_v7, 19)
    bootable = recovery.prepare_startup()
    assert bootable.snapshot == retained_v7
    assert path.read_bytes() == retained_bytes

    candidate, _source = _capture_v8(failing_store, populated=True)
    event = _event("failed-first-v8-publication", 20)
    _start_event(journal, event)
    with pytest.raises(AgentStateSaveError) as error:
        recovery.commit_internal_candidate(event, retained_v7, candidate)

    assert error.value.stage is AgentStateSaveStage.ATOMIC_REPLACE
    assert error.value.published is False
    assert path.read_bytes() == retained_bytes
    assert failing_store.load() == retained_v7
    assert wal.reconstruct(sequence=20) == candidate
    # Prepared/WAL evidence is not a committed v8 authority. Restart must
    # recover the retained v7 state rather than replay or promote that tail.
    restarted = StateRecoveryCoordinator(_store(path), journal, wal).prepare_startup()
    assert restarted.snapshot == retained_v7
    assert _store(path).load() == retained_v7
    assert path.read_bytes() == retained_bytes


def test_true_rollback_from_v8_to_v7_clears_r13_restore_ports(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    captured_v7 = store.capture(_Loop(), sequence=19)
    assert isinstance(captured_v7, AgentStateSnapshotV7)
    retained_v7 = captured_v7.model_copy(
        update={
            "emotion_state": captured_v7.emotion_state.model_copy(
                update={"valence": 0.17}
            )
        }
    )
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, retained_v7)
    newer_v8, _source = _capture_v8(store, populated=True)
    _commit(
        recovery,
        journal,
        _event("v8-then-v7-true-rollback", 20),
        retained_v7,
        newer_v8,
    )

    corrupt_generation = _corrupt_active_transition(wal)
    anchor = wal.inspect_boot_anchor()
    generation = wal.root / "generations" / f"{anchor.generation_id}.jsonl"
    # The deliberately corrupted bytes remain in the retired generation.
    assert generation.read_bytes() == corrupt_generation
    store.path.unlink()

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert rolled_back.snapshot == retained_v7
    assert rolled_back.true_rollback_performed
    assert rolled_back.external_reconciliation_required
    assert wal.inspect().records[0].baseline_snapshot == retained_v7
    assert store.load() == retained_v7

    target = _R13Loop(_domains(populated=True))
    store.restore_into(target, rolled_back.snapshot)
    assert target.r13_state() == _domains(populated=False)


def test_true_rollback_to_older_v8_restores_exact_r13_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    older_v8, _older_source = _capture_v8(store, populated=True)
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, older_v8)
    newer_v8 = store.capture(_R13Loop(_domains(populated=False)), sequence=21)
    assert isinstance(newer_v8, AgentStateSnapshotV8)
    _commit(
        recovery,
        journal,
        _event("newer-v8-then-older-v8-rollback", 21),
        older_v8,
        newer_v8,
    )

    corrupt_generation = _corrupt_active_transition(wal)
    anchor = wal.inspect_boot_anchor()
    generation = wal.root / "generations" / f"{anchor.generation_id}.jsonl"
    assert generation.read_bytes() == corrupt_generation
    store.path.unlink()

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert rolled_back.snapshot == older_v8
    assert isinstance(rolled_back.snapshot, AgentStateSnapshotV8)
    assert rolled_back.true_rollback_performed
    assert rolled_back.external_reconciliation_required
    assert wal.inspect().records[0].baseline_snapshot == older_v8

    target = _R13Loop(_domains(populated=False))
    store.restore_into(target, rolled_back.snapshot)
    assert target.r13_state() == _domains(populated=True)


def test_committed_v8_crash_reconstructs_without_policy_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "agent_state.json")
    retained_v7 = store.capture(_Loop(), sequence=19)
    assert isinstance(retained_v7, AgentStateSnapshotV7)
    recovery, journal, wal = _boot_from_snapshot(tmp_path, store, retained_v7)
    candidate, expected_source = _capture_v8(store, populated=True)
    event = _event("committed-v8-before-crash", 20)
    _start_event(journal, event)

    # Internal state and the WAL are durable; terminal Journal completion is
    # intentionally omitted to model a crash at the committed boundary.
    evidence = recovery.commit_internal_candidate(event, retained_v7, candidate)
    assert evidence.snapshot_hash == store.snapshot_hash(candidate)
    assert journal.inspect().records[-1].state_hash_after == evidence.snapshot_hash

    def forbidden_policy_replay(*_args: object, **_kwargs: object) -> None:
        pytest.fail("R13 policy operation was replayed during crash recovery")

    for domain, operations in (
        (MotivationSystem, ("apply_evidence", "ingest_candidate")),
        (GoalSystem, ("ingest_proposal", "adopt", "defer", "resume", "abandon")),
        (
            CommitmentSystem,
            ("ingest_proposal", "accept", "release", "renegotiate"),
        ),
    ):
        for operation in operations:
            monkeypatch.setattr(domain, operation, forbidden_policy_replay)

    restarted = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert restarted.snapshot == candidate
    assert isinstance(restarted.snapshot, AgentStateSnapshotV8)
    assert restarted.snapshot_hash == evidence.snapshot_hash
    assert restarted.processing_high_water == 20
    assert wal.reconstruct(sequence=20) == candidate
    target = _R13Loop(_domains(populated=False))
    store.restore_into(target, restarted.snapshot)
    assert target.r13_state() == expected_source.r13_state()


def test_wal_rejects_mutated_nested_r13_table_before_append(tmp_path: Path) -> None:
    store = _store(tmp_path / "agent_state.json")
    prior, _prior_source = _capture_v8(store, populated=True)
    candidate, _candidate_source = _capture_v8(store, populated=True)
    candidate = candidate.model_copy(update={"last_processed_event_sequence": 21})
    wal = StateWAL(tmp_path / "wal")
    manifest = wal.bootstrap(prior, 20)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    before = wal.inspect()
    before_bytes = generation.read_bytes()

    first_row = candidate.r13_state.goal.nodes[0]
    assert isinstance(first_row, list)
    first_row[0] = "UnregisteredR13Node"

    with pytest.raises(StateWALError):
        wal.append_transition(
            event_id=uuid5(NAMESPACE_URL, "mutated-r13-table-append"),
            event_type="state.transition",
            event_source="test",
            processing_sequence=21,
            prior_snapshot=prior,
            candidate_snapshot=candidate,
        )

    after = wal.inspect()
    assert after.records == before.records
    assert after.record_hashes == before.record_hashes
    assert generation.read_bytes() == before_bytes


def test_malformed_persisted_v8_snapshot_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _source = _capture_v8(store, populated=True)
    store.save(snapshot)
    persisted = json.loads(store.path.read_bytes())
    persisted["r13_state"]["goal"]["nodes"][0][0] = "UnregisteredR13Node"
    store.path.write_text(json.dumps(persisted, separators=(",", ":")))

    with pytest.raises(AgentStateLoadError, match="schema is invalid"):
        store.load()
