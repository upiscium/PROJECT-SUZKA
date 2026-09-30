"""Focused AgentState v7 capacity and compatibility checks."""

from datetime import UTC, datetime
from pathlib import Path

import suzka.runtime.agent_state as agent_state_module
import pytest
from pydantic import ValidationError
from suzka.belief import BeliefSystem
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
from suzka.runtime import (
    AgentStateSaveError,
    AgentStateSaveStage,
    AGENT_STATE_FUTURE_STATE_RESERVE_BYTES,
    AGENT_STATE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V7_SCHEMA_FIELD_MAX_BYTES,
    AGENT_STATE_V7_SCHEMA_MAX_SERIALIZED_BYTES,
    AgentStateSnapshotV6,
    AgentStateSnapshotV7,
    AgentStateStore,
    AppraisalStateSnapshot,
    BeliefSystemStateSnapshot,
    ContextStateSnapshot,
    EmotionStateSnapshot,
    WorkingMemoryItemSnapshot,
    ValueSystemStateSnapshot,
    WorkingMemorySnapshot,
    default_agent_state_snapshot,
    project_agent_state_schema_max_bytes,
)


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
R13_U1_REVIEWED_MAX_FIELD_BYTES = 18_759_305


def _empty_v6_snapshot() -> AgentStateSnapshotV6:
    belief = agent_state_module._belief_state_snapshot(BeliefSystem().snapshot())
    return AgentStateSnapshotV6(
        saved_at=NOW,
        last_processed_event_sequence=0,
        emotion_state=EmotionStateSnapshot(
            valence=0.0,
            arousal=0.0,
            optimal_loss=1.0,
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
        context_state=ContextStateSnapshot(
            revision=0,
            current_context_id=None,
            frames=(),
            interlocutor_bindings=(),
        ),
        appraisal_state=AppraisalStateSnapshot(
            calibration_entries=(),
            last_emotion_update_at=None,
        ),
        value_state=ValueSystemStateSnapshot(
            schema_version=1,
            values=(),
            conflicts=(),
            histories=(),
            evidence_ledgers=(),
        ),
        belief_state=BeliefSystemStateSnapshot(
            schema_version=belief.schema_version,
            records=belief.records,
            authority_digest=belief.authority_digest,
        ),
    )


def test_v7_default_and_capacity_projection_fit_the_hard_bound() -> None:
    snapshot = default_agent_state_snapshot(1.0, saved_at=NOW)
    payload = AgentStateStore._canonical_bytes(snapshot)

    assert snapshot.schema_version == 7
    assert set(AGENT_STATE_V7_SCHEMA_FIELD_MAX_BYTES) == set(
        AgentStateSnapshotV7.model_fields
    )
    assert AGENT_STATE_V7_SCHEMA_FIELD_MAX_BYTES == {
        "appraisal_state": 11_536,
        "belief_state": 47_541_622,
        "context_state": 18_473_172,
        "emotion_state": 114,
        "last_processed_event_sequence": 19,
        "saved_at": 29,
        "schema_version": 1,
        "value_state": 18_054_672,
        "working_memory": 1_744_929,
    }
    assert sum(AGENT_STATE_V7_SCHEMA_FIELD_MAX_BYTES.values()) == 85_826_094
    assert AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES == 85_826_260
    assert len(payload) <= AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES
    assert (
        AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES
        + AGENT_STATE_FUTURE_STATE_RESERVE_BYTES
        == AGENT_STATE_V7_SCHEMA_MAX_SERIALIZED_BYTES
    )
    assert AGENT_STATE_V7_SCHEMA_MAX_SERIALIZED_BYTES <= (
        AGENT_STATE_MAX_SERIALIZED_BYTES
    )
    assert project_agent_state_schema_max_bytes() == (
        AGENT_STATE_V7_SCHEMA_MAX_SERIALIZED_BYTES
    )


def test_r13_v8_projection_keeps_the_r14_future_reserve() -> None:
    assert R13_U1_REVIEWED_MAX_FIELD_BYTES == 18_759_305
    v6_max = agent_state_module.AGENT_STATE_V6_SCHEMA_MAX_SERIALIZED_BYTES
    assert v6_max == 151_284_436
    assert v6_max > AGENT_STATE_MAX_SERIALIZED_BYTES
    projected = project_agent_state_schema_max_bytes(
        schema_version=8,
        added_field_maxima={
            "motivation_state": R13_U1_REVIEWED_MAX_FIELD_BYTES,
        },
    )

    assert projected == 121_362_801
    projected_without_reserve = projected - AGENT_STATE_FUTURE_STATE_RESERVE_BYTES
    assert projected_without_reserve == 104_585_585
    assert projected <= AGENT_STATE_MAX_SERIALIZED_BYTES
    assert AGENT_STATE_MAX_SERIALIZED_BYTES - projected_without_reserve >= (
        AGENT_STATE_FUTURE_STATE_RESERVE_BYTES
    )
    assert AGENT_STATE_MAX_SERIALIZED_BYTES - projected == 12_854_927


def test_v6_load_is_byte_preserving_and_does_not_eagerly_migrate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = AgentStateStore(path, baseline_surprisal=1.0, clock=lambda: NOW)
    snapshot = _empty_v6_snapshot()
    store.save(snapshot)
    before = path.read_bytes()

    loaded = store.load()

    assert isinstance(loaded, AgentStateSnapshotV6)
    assert path.read_bytes() == before


def test_v6_file_survives_failed_normal_v7_publication(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    legacy_store = AgentStateStore(path, baseline_surprisal=1.0, clock=lambda: NOW)
    legacy_store.save(_empty_v6_snapshot())
    before = path.read_bytes()

    def fail_before_replace(stage: AgentStateSaveStage) -> None:
        if stage is AgentStateSaveStage.ATOMIC_REPLACE:
            raise RuntimeError("injected pre-replacement failure")

    migrating_store = AgentStateStore(
        path,
        baseline_surprisal=1.0,
        clock=lambda: NOW,
        save_stage_hook=fail_before_replace,
    )
    candidate = default_agent_state_snapshot(1.0, saved_at=NOW)
    with pytest.raises(AgentStateSaveError):
        migrating_store.save(candidate)

    assert path.read_bytes() == before


def test_future_projection_is_not_silently_clamped_to_the_hard_bound() -> None:
    projected = project_agent_state_schema_max_bytes(
        added_field_maxima={
            "future_state": AGENT_STATE_MAX_SERIALIZED_BYTES,
        }
    )

    assert projected > AGENT_STATE_MAX_SERIALIZED_BYTES


def test_agent_state_snapshots_reject_over_bound_event_sequence() -> None:
    snapshots = (
        _empty_v6_snapshot(),
        default_agent_state_snapshot(1.0, saved_at=NOW),
    )
    for snapshot in snapshots:
        raw = snapshot.model_dump(mode="python")
        raw["last_processed_event_sequence"] = MAX_PERSISTED_EVENT_SEQUENCE + 1

        with pytest.raises(ValidationError):
            agent_state_module.validate_compatible_agent_state_snapshot(raw)


def test_working_memory_snapshot_enforces_projected_item_and_revision_bounds() -> None:
    with pytest.raises(ValidationError):
        WorkingMemorySnapshot(
            revision=MAX_PERSISTED_REVISION + 1,
            items=(),
        )

    with pytest.raises(ValidationError):
        WorkingMemoryItemSnapshot(
            item_id="wm-invalid-revision",
            source_kind="episodic",
            source_id="episode-1",
            activation=0.5,
            salience=0.5,
            retention_reason="recent",
            created_revision=MAX_PERSISTED_REVISION + 1,
            last_activated_revision=MAX_PERSISTED_REVISION + 1,
        )

    item_id = agent_state_module.working_memory_item_id(
        agent_state_module.WorkingMemorySourceKind.EPISODIC, "episode-1"
    )
    item = WorkingMemoryItemSnapshot(
        item_id=item_id,
        source_kind="episodic",
        source_id="episode-1",
        activation=0.5,
        salience=0.5,
        retention_reason="recent",
        created_revision=0,
        last_activated_revision=0,
    )
    with pytest.raises(ValidationError):
        WorkingMemorySnapshot(
            revision=0,
            items=(item,) * (agent_state_module.MAX_ITEM_CAPACITY + 1),
        )
