"""U5 integration tests for version-8 AgentState R13 authority continuity."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
from pydantic import ValidationError

from suzka.belief import BeliefSystem, BeliefSystemSnapshot
from suzka.body import EmotionEngineAllostasis, EmotionState, EmotionTemporalState
from suzka.cognition.surprisal_calculator import LossCalibration
from suzka.identity.value_system import ValueSystem
from suzka.motivation.commitment_system import CommitmentSystem, CommitmentSystemSnapshot
from suzka.motivation.goal_system import GoalSystem, GoalSystemSnapshot
from suzka.motivation.system import MotivationSystem, MotivationSystemSnapshot
from suzka.runtime.agent_state import (
    AgentStateLoadError,
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshotV7,
    AgentStateSnapshotV8,
    AgentStateStore,
    validate_compatible_agent_state_snapshot,
)
from suzka.runtime.context import ContextRegistry
from suzka.runtime.r13_codec import R13StateSnapshot
from suzka.runtime.working_memory import WorkingMemory
from test_r13_codec import (
    _commitment_snapshot,
    _goal_snapshot,
    _motivation_snapshot,
)


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _domains(*, populated: bool) -> tuple[
    MotivationSystemSnapshot,
    GoalSystemSnapshot,
    CommitmentSystemSnapshot,
]:
    if populated:
        return _motivation_snapshot(), _goal_snapshot(), _commitment_snapshot()
    empty = R13StateSnapshot.empty()
    motivation = empty.motivation.restore()
    goal = empty.goal.restore()
    commitment = empty.commitment.restore()
    assert isinstance(motivation, MotivationSystemSnapshot)
    assert isinstance(goal, GoalSystemSnapshot)
    assert isinstance(commitment, CommitmentSystemSnapshot)
    return motivation, goal, commitment


class _Loop:
    agent_state_ports: object | None

    def __init__(self) -> None:
        self.agent_state_ports = None
        self.emotion_engine = EmotionEngineAllostasis(
            EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0)
        )
        self.working_memory = WorkingMemory(
            item_capacity=32,
            projection_max_bytes=2048,
        )
        self.context_registry = ContextRegistry(clock=lambda: NOW)
        self.loss_calibration = LossCalibration(
            ("model." + "a" * 64,),
            initial_baseline=1.0, initial_scale=1.0, minimum_scale=0.1
        )
        self.value_system = ValueSystem()
        self._belief: BeliefSystemSnapshot = BeliefSystem().snapshot()

    def export_belief_state(self) -> BeliefSystemSnapshot:
        return self._belief

    def restore_belief_state(self, snapshot: BeliefSystemSnapshot) -> None:
        self._belief = snapshot


class _R13Loop(_Loop):
    def __init__(
        self,
        domains: tuple[
            MotivationSystemSnapshot,
            GoalSystemSnapshot,
            CommitmentSystemSnapshot,
        ],
    ) -> None:
        super().__init__()
        self._motivation: MotivationSystemSnapshot = domains[0]
        self._goal: GoalSystemSnapshot = domains[1]
        self._commitment: CommitmentSystemSnapshot = domains[2]
        self.restore_calls: list[str] = []
        self.fail_restore: str | None = None

    def export_motivation_state(self) -> MotivationSystemSnapshot:
        return self._motivation

    def restore_motivation_state(self, snapshot: MotivationSystemSnapshot) -> None:
        self._restore("motivation", snapshot)

    def export_goal_state(self) -> GoalSystemSnapshot:
        return self._goal

    def restore_goal_state(self, snapshot: GoalSystemSnapshot) -> None:
        self._restore("goal", snapshot)

    def export_commitment_state(self) -> CommitmentSystemSnapshot:
        return self._commitment

    def restore_commitment_state(self, snapshot: CommitmentSystemSnapshot) -> None:
        self._restore("commitment", snapshot)

    def _restore(self, domain: str, snapshot: object) -> None:
        self.restore_calls.append(domain)
        setattr(self, f"_{domain}", snapshot)
        if self.fail_restore == domain:
            self.fail_restore = None
            raise RuntimeError(f"injected {domain} restore failure")

    def r13_state(
        self,
    ) -> tuple[
        MotivationSystemSnapshot,
        GoalSystemSnapshot,
        CommitmentSystemSnapshot,
    ]:
        return self._motivation, self._goal, self._commitment


class _PartialR13Loop(_Loop):
    def export_motivation_state(self) -> MotivationSystemSnapshot:
        return _domains(populated=False)[0]

    def restore_motivation_state(self, snapshot: MotivationSystemSnapshot) -> None:
        del snapshot


class _AgentStatePorts:
    def __init__(self, owner: _R13Loop) -> None:
        self.motivation_state_port = owner
        self.goal_state_port = owner
        self.commitment_state_port = owner


class _FailingTemporalEmotionEngine(EmotionEngineAllostasis):
    def __init__(self, state: EmotionState) -> None:
        self._fail_temporal = False
        self._temporal_state = EmotionTemporalState()
        super().__init__(state)

    @property
    def temporal_state(self) -> EmotionTemporalState:
        return self._temporal_state

    @temporal_state.setter
    def temporal_state(self, value: EmotionTemporalState) -> None:
        if self._fail_temporal:
            self._fail_temporal = False
            raise RuntimeError("injected temporal restore failure")
        self._temporal_state = value


def _store(
    path: Path,
    *,
    hook: Callable[[AgentStateSaveStage], None] | None = None,
) -> AgentStateStore:
    return AgentStateStore(
        path,
        baseline_surprisal=1.0,
        clock=lambda: NOW,
        save_stage_hook=hook,
    )


def _capture_v8(
    store: AgentStateStore,
    *,
    populated: bool,
) -> tuple[AgentStateSnapshotV8, _R13Loop]:
    source = _R13Loop(_domains(populated=populated))
    snapshot = store.capture(source, sequence=20)
    assert isinstance(snapshot, AgentStateSnapshotV8)
    return snapshot, source


@pytest.mark.parametrize("populated", [False, True])
def test_v8_round_trip_preserves_complete_empty_and_nontrivial_r13_graphs(
    tmp_path: Path, populated: bool
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, source = _capture_v8(store, populated=populated)
    if populated:
        motivation = snapshot.r13_state.motivation.restore()
        goal = snapshot.r13_state.goal.restore()
        commitment = snapshot.r13_state.commitment.restore()
        assert isinstance(motivation, MotivationSystemSnapshot)
        assert isinstance(goal, GoalSystemSnapshot)
        assert isinstance(commitment, CommitmentSystemSnapshot)
        assert motivation.candidates
        assert goal.records[0].history_anchor is not None
        assert goal.records[0].subject_transition_proofs
        assert {record.lifecycle.value for record in commitment.records} == {
            "released",
            "renegotiated",
        }

    store.save(snapshot)
    target = _R13Loop(_domains(populated=False))
    loaded = store.load()
    assert isinstance(loaded, AgentStateSnapshotV8)
    store.restore_into(target, loaded)

    assert target.r13_state() == source.r13_state()
    assert target.restore_calls == ["motivation", "goal", "commitment"]


def test_v7_belief_capture_stays_v7_and_r13_capture_upgrades_to_v8(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")

    legacy = store.capture(_Loop(), sequence=3)
    assert isinstance(legacy, AgentStateSnapshotV7)
    assert legacy.schema_version == 7

    target = _R13Loop(_domains(populated=True))
    store.restore_into(target, legacy)
    assert target.r13_state() == _domains(populated=False)

    current = store.capture(target, sequence=4)
    assert isinstance(current, AgentStateSnapshotV8)
    assert current.schema_version == 8


def test_v8_capture_and_restore_use_optional_agent_state_port_fields(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    source = _Loop()
    source_ports = _R13Loop(_domains(populated=True))
    source.agent_state_ports = _AgentStatePorts(source_ports)
    snapshot = store.capture(source, sequence=20)
    assert isinstance(snapshot, AgentStateSnapshotV8)

    target = _Loop()
    target_ports = _R13Loop(_domains(populated=False))
    target.agent_state_ports = _AgentStatePorts(target_ports)
    store.restore_into(target, snapshot)

    assert target_ports.r13_state() == source_ports.r13_state()


def test_persisted_v8_restores_actual_domain_ports_exactly(tmp_path: Path) -> None:
    source = _Loop()
    motivation, goal, commitment = MotivationSystem(), GoalSystem(), CommitmentSystem()
    snapshots = _domains(populated=True)
    motivation.restore_motivation_state(snapshots[0])
    goal.restore_goal_state(snapshots[1])
    commitment.restore_commitment_state(snapshots[2])
    source.agent_state_ports = SimpleNamespace(
        motivation_state_port=motivation, goal_state_port=goal, commitment_state_port=commitment,
    )
    store = _store(tmp_path / "agent_state.json")
    captured = store.capture(source, sequence=20)
    store.save(captured)

    target = _Loop()
    restored_motivation, restored_goal, restored_commitment = (
        MotivationSystem(), GoalSystem(), CommitmentSystem()
    )
    target.agent_state_ports = SimpleNamespace(
        motivation_state_port=restored_motivation,
        goal_state_port=restored_goal,
        commitment_state_port=restored_commitment,
    )
    store.restore_into(target, store.load())
    actual = (restored_motivation.snapshot(), restored_goal.snapshot(), restored_commitment.snapshot())
    assert actual == snapshots
    assert tuple(item.authority_digest for item in actual) == tuple(
        item.authority_digest for item in snapshots
    )


def test_partial_r13_capture_topology_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(AgentStateSaveError) as error:
        _store(tmp_path / "agent_state.json").capture(_PartialR13Loop(), sequence=1)

    assert error.value.stage is AgentStateSaveStage.CAPTURE
    assert error.value.published is False


def test_v8_typed_models_and_mutable_nested_tables_are_revalidated(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _source = _capture_v8(store, populated=False)
    raw = snapshot.model_dump(mode="python")
    raw.pop("r13_state")
    with pytest.raises(ValidationError):
        AgentStateSnapshotV8.model_validate(raw)

    graph = snapshot.r13_state.motivation
    first_row = graph.nodes[0]
    assert isinstance(first_row, list)
    first_row[0] = "UnregisteredR13Node"
    with pytest.raises(ValidationError):
        validate_compatible_agent_state_snapshot(snapshot)
    with pytest.raises(AgentStateSaveError):
        store.canonical_bytes(snapshot)


def test_v8_events_must_precede_root_sequence_and_saved_time(tmp_path: Path) -> None:
    snapshot, _ = _capture_v8(_store(tmp_path / "agent_state.json"), populated=True)
    raw = snapshot.model_dump(mode="python")
    raw["last_processed_event_sequence"] = 1
    with pytest.raises(ValidationError, match="future of its AgentState"):
        AgentStateSnapshotV8.model_validate(raw)
    raw = snapshot.model_dump(mode="python")
    raw["saved_at"] = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="future of its AgentState"):
        AgentStateSnapshotV8.model_validate(raw)


def test_empty_v8_without_r13_ports_retains_legacy_capture_path(tmp_path: Path) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _ = _capture_v8(store, populated=False)
    target = _Loop()
    store.restore_into(target, snapshot)
    captured = store.capture(target, sequence=21)
    assert isinstance(captured, AgentStateSnapshotV7)
    target.agent_state_ports = type("EmptyPorts", (), {
        "motivation_state_port": None, "goal_state_port": None, "commitment_state_port": None,
    })()
    assert isinstance(store.capture(target, sequence=21), AgentStateSnapshotV7)


def test_v8_nonempty_state_without_all_ports_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _source = _capture_v8(store, populated=True)
    target = _Loop()
    before_emotion = target.emotion_engine.state

    with pytest.raises(AgentStateLoadError, match="all R13 state ports"):
        store.restore_into(target, snapshot)

    assert target.emotion_engine.state == before_emotion


def test_v7_restore_clears_available_r13_ports(tmp_path: Path) -> None:
    store = _store(tmp_path / "agent_state.json")
    legacy = store.capture(_Loop(), sequence=3)
    target = _R13Loop(_domains(populated=True))

    store.restore_into(target, legacy)

    assert target.r13_state() == _domains(populated=False)


def test_first_v8_save_failure_leaves_existing_v7_bytes_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = _store(path)
    legacy = store.capture(_Loop(), sequence=3)
    assert isinstance(legacy, AgentStateSnapshotV7)
    store.save(legacy)
    before = path.read_bytes()

    def fail_before_replace(stage: AgentStateSaveStage) -> None:
        if stage is AgentStateSaveStage.ATOMIC_REPLACE:
            raise RuntimeError("injected v8 publication failure")

    v8_store = _store(path, hook=fail_before_replace)
    candidate, _source = _capture_v8(v8_store, populated=True)
    with pytest.raises(AgentStateSaveError) as error:
        v8_store.save(candidate)

    assert error.value.stage is AgentStateSaveStage.ATOMIC_REPLACE
    assert error.value.published is False
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("failure_domain", "expected_calls"),
    [
        (
            "goal",
            ["motivation", "goal", "goal", "motivation"],
        ),
        (
            "commitment",
            [
                "motivation",
                "goal",
                "commitment",
                "commitment",
                "goal",
                "motivation",
            ],
        ),
    ],
)
def test_r13_restore_failures_roll_back_each_attempted_authority(
    tmp_path: Path,
    failure_domain: str,
    expected_calls: list[str],
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _source = _capture_v8(store, populated=True)
    target = _R13Loop(_domains(populated=False))
    before = target.r13_state()
    target.fail_restore = failure_domain

    with pytest.raises(AgentStateLoadError, match="restore failed"):
        store.restore_into(target, snapshot)

    assert target.r13_state() == before
    assert target.restore_calls == expected_calls


def test_later_upstream_restore_failure_rolls_back_all_r13_authorities(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _source = _capture_v8(store, populated=True)
    target = _R13Loop(_domains(populated=False))
    before_r13 = target.r13_state()
    before_emotion = target.emotion_engine.state
    failing_engine = _FailingTemporalEmotionEngine(before_emotion)
    target.emotion_engine = failing_engine
    before_temporal = EmotionTemporalState(datetime(2026, 1, 1, tzinfo=UTC))
    failing_engine.temporal_state = before_temporal
    failing_engine._fail_temporal = True

    with pytest.raises(AgentStateLoadError, match="restore failed"):
        store.restore_into(target, snapshot)

    assert target.r13_state() == before_r13
    assert target.restore_calls == [
        "motivation",
        "goal",
        "commitment",
        "commitment",
        "goal",
        "motivation",
    ]
    assert target.emotion_engine.state == before_emotion
    assert target.emotion_engine.temporal_state == before_temporal
