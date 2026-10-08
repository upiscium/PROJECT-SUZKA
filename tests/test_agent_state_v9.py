"""U4 AgentState v9 Attention continuity and recovery integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from suzka.attention.adapters import project_goal, project_working_memory
from suzka.attention.bounds import maximum_attention_continuity_fixture
from suzka.attention.contracts import AttentionContinuity, AttentionEvent
from suzka.attention.system import AttentionSystem
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE
from suzka.runtime.agent_state import (
    AGENT_STATE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_SCHEMA_VERSION_V8,
    AGENT_STATE_V8_BASE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V8_SCHEMA_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES,
    AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES,
    CURRENT_AGENT_STATE_SCHEMA_VERSION,
    AgentStateLoadError,
    AgentStateSaveError,
    AgentStateSnapshot,
    AgentStateSnapshotV7,
    AgentStateSnapshotV8,
    AgentStateSnapshotV9,
    AgentStateStore,
    project_agent_state_schema_max_bytes,
    validate_compatible_agent_state_snapshot,
)
from suzka.runtime.attention_state_codec import (
    AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES,
    preflight_attention_json,
)
from suzka.runtime.attention_state_port import AttentionStatePort
from suzka.runtime.working_memory import WorkingMemorySourceKind
from test_agent_state_v8 import (
    NOW,
    _Loop,
    _R13Loop,
    _capture_v8,
    _domains,
    _store,
)


@pytest.fixture(autouse=True)
def _isolate_user_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUZKA_CONFIG_PATH", str(tmp_path / "config.yaml"))


def _event(sequence: int, event_id: str = "attention:agent-state") -> AttentionEvent:
    return AttentionEvent(event_id, sequence, NOW)


def _continuity(sequence: int = 20) -> AttentionContinuity:
    return AttentionSystem().refresh((), _event(sequence)).snapshot


def _v9_from_v8(
    base: AgentStateSnapshotV8,
    attention: AttentionContinuity,
    *,
    sequence: int = 20,
    saved_at: datetime = NOW,
) -> AgentStateSnapshotV9:
    raw = base.model_dump(mode="python")
    raw["schema_version"] = 9
    raw["last_processed_event_sequence"] = sequence
    raw["saved_at"] = saved_at
    raw["attention_state"] = attention
    return AgentStateSnapshotV9.model_validate(raw)


def _attach_attention(
    main_loop: _Loop,
    attention: AttentionSystem,
    r13_owner: _R13Loop | None = None,
) -> tuple[_Loop, _R13Loop]:
    owner = r13_owner or _R13Loop(_domains(populated=False))
    main_loop.agent_state_ports = SimpleNamespace(
        motivation_state_port=owner,
        goal_state_port=owner,
        commitment_state_port=owner,
        attention_state_port=AttentionStatePort(attention),
    )
    return main_loop, owner


def _captured_v9(
    tmp_path: Path,
    *,
    source_memory: bool = False,
) -> tuple[
    AgentStateStore,
    AgentStateSnapshotV9,
    _Loop,
    _R13Loop,
    AttentionSystem,
]:
    store = _store(tmp_path / "agent_state.json")
    main_loop = _Loop()
    if source_memory:
        main_loop.working_memory.admit(
            WorkingMemorySourceKind.EPISODIC,
            "agent-state-attention-member",
            activation=0.8,
            salience=0.6,
        )
    r13_owner = _R13Loop(_domains(populated=source_memory))
    event = _event(20, "attention:agent-state:capture")
    projections = []
    if source_memory:
        projections.extend(
            (
                project_goal(r13_owner.export_goal_state().records[0], event=event),
                project_working_memory(
                    main_loop.working_memory.items[0],
                    revision=main_loop.working_memory.revision,
                    event=event,
                ),
            )
        )
    attention = AttentionSystem()
    attention.refresh(tuple(projections), event)
    main_loop, r13_owner = _attach_attention(main_loop, attention, r13_owner)
    snapshot = store.capture(main_loop, sequence=20)
    assert isinstance(snapshot, AgentStateSnapshotV9)
    return store, snapshot, main_loop, r13_owner, attention


def test_v9_requires_a_closed_attention_field_and_root_event_fences(
    tmp_path: Path,
) -> None:
    base, _owner = _capture_v8(_store(tmp_path / "base.json"), populated=False)
    snapshot = _v9_from_v8(base, _continuity())
    raw = snapshot.model_dump(mode="python")

    missing = dict(raw)
    del missing["attention_state"]
    with pytest.raises(ValidationError):
        AgentStateSnapshotV9.model_validate(missing)

    extra = dict(raw)
    extra["attention_ephemeral_assessment"] = {}
    with pytest.raises(ValidationError):
        AgentStateSnapshotV9.model_validate(extra)

    future_sequence = dict(raw)
    future_sequence["last_processed_event_sequence"] = 19
    with pytest.raises(ValidationError, match="future of its AgentState"):
        AgentStateSnapshotV9.model_validate(future_sequence)

    future_time = dict(raw)
    future_time["saved_at"] = NOW - timedelta(seconds=1)
    with pytest.raises(ValidationError, match="future of its AgentState"):
        AgentStateSnapshotV9.model_validate(future_time)


@pytest.mark.parametrize(
    "schema_version",
    [
        pytest.param(9.0, id="float-nine"),
        pytest.param(9.0000001, id="float-above-nine"),
        pytest.param(8.99999999999999999999, id="below-nine-rounded-to-nine"),
        pytest.param(9e0, id="exponent-float-nine"),
        pytest.param("9", id="string-nine"),
        pytest.param(True, id="boolean"),
    ],
)
def test_v9_schema_version_rejects_non_exact_integer_inputs(
    tmp_path: Path,
    schema_version: object,
) -> None:
    base, _owner = _capture_v8(_store(tmp_path / "base.json"), populated=False)
    valid = _v9_from_v8(base, AttentionContinuity.bootstrap())
    raw = valid.model_dump(mode="python")
    raw["schema_version"] = schema_version

    with pytest.raises(ValidationError):
        AgentStateSnapshotV9.model_validate(raw)
    with pytest.raises(ValidationError):
        validate_compatible_agent_state_snapshot(raw)


def test_rounded_raw_v9_version_with_duplicate_attention_key_is_rejected(
    tmp_path: Path,
) -> None:
    store, snapshot, _source, _owner, _attention = _captured_v9(tmp_path)
    payload = store.canonical_bytes(snapshot)
    root_version = b'"schema_version":9,"value_state":'
    assert payload.count(root_version) == 1
    payload = payload.replace(
        root_version,
        b'"schema_version":9.0000000000000000001,"value_state":',
        1,
    )
    policy_version = b'"policy_version":1'
    assert payload.count(policy_version) == 1
    payload = payload.replace(
        policy_version,
        b'"policy_version":0,"policy_version":1',
        1,
    )
    store.path.write_bytes(payload)

    with pytest.raises(AgentStateLoadError):
        store.load()

    assert store.path.read_bytes() == payload


def test_v9_alias_round_trip_uses_canonical_attention_and_requires_owner(
    tmp_path: Path,
) -> None:
    store, snapshot, _source, source_r13, source_attention = _captured_v9(
        tmp_path, source_memory=True
    )
    payload = store.canonical_bytes(snapshot)
    persisted = json.loads(payload)
    assert isinstance(persisted["attention_state"], dict)
    assert persisted["attention_state"] == snapshot.attention_state.canonical_value()
    assert AgentStateSnapshot is AgentStateSnapshotV9

    store.save(snapshot)
    loaded = store.load()
    assert isinstance(loaded, AgentStateSnapshotV9)

    target_attention = AttentionSystem()
    target = _Loop()
    target, target_r13 = _attach_attention(target, target_attention)
    source_view = source_attention.selected_view()
    assert source_view.competition is not None
    assert source_view.prompt is not None
    store.restore_into(target, loaded)
    assert target_attention.snapshot() == source_attention.snapshot()
    assert target_r13.r13_state() == source_r13.r13_state()
    # Restores retain durable focus references, but rebuild no transient policy
    # competition or prompt assessment from the saved continuity.
    restored_view = target_attention.selected_view()
    assert restored_view.competition is None
    assert restored_view.prompt is None

    without_port = _Loop()
    no_attention_r13 = _R13Loop(_domains(populated=False))
    without_port.agent_state_ports = SimpleNamespace(
        motivation_state_port=no_attention_r13,
        goal_state_port=no_attention_r13,
        commitment_state_port=no_attention_r13,
    )
    with pytest.raises(AgentStateLoadError, match="Attention state port"):
        store.restore_into(without_port, loaded)


def test_v9_requires_attention_port_even_for_empty_bootstrap_state(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    base, _owner = _capture_v8(store, populated=False)
    snapshot = _v9_from_v8(base, AttentionContinuity.bootstrap())
    target = _Loop()
    r13_owner = _R13Loop(_domains(populated=False))
    target.agent_state_ports = SimpleNamespace(
        motivation_state_port=r13_owner,
        goal_state_port=r13_owner,
        commitment_state_port=r13_owner,
    )

    with pytest.raises(AgentStateLoadError, match="Attention state port"):
        store.restore_into(target, snapshot)


def test_attention_capture_is_opt_in_and_requires_the_complete_legacy_topology(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "without-port.json")
    v8, _source = _capture_v8(store, populated=False)
    assert isinstance(v8, AgentStateSnapshotV8)
    assert v8.schema_version == AGENT_STATE_SCHEMA_VERSION_V8 == 8

    partial = _Loop()
    partial.attention_state_port = AttentionStatePort(AttentionSystem())
    with pytest.raises(AgentStateSaveError) as failure:
        store.capture(partial, sequence=20)
    assert failure.value.published is False

    duck_port = _Loop()
    duck_port.attention_state_port = SimpleNamespace(
        export_attention_state=lambda: AttentionContinuity.bootstrap()
    )
    with pytest.raises(AgentStateSaveError):
        store.capture(duck_port, sequence=20)

    partial_r13 = _Loop()
    partial_owner = _R13Loop(_domains(populated=False))
    partial_r13.agent_state_ports = SimpleNamespace(
        motivation_state_port=partial_owner,
        goal_state_port=None,
        commitment_state_port=None,
        attention_state_port=AttentionStatePort(AttentionSystem()),
    )
    with pytest.raises(AgentStateSaveError):
        store.capture(partial_r13, sequence=20)

    missing_belief = _Loop()
    missing_belief.export_belief_state = None
    full_owner = _R13Loop(_domains(populated=False))
    missing_belief.agent_state_ports = SimpleNamespace(
        motivation_state_port=full_owner,
        goal_state_port=full_owner,
        commitment_state_port=full_owner,
        attention_state_port=AttentionStatePort(AttentionSystem()),
    )
    with pytest.raises(AgentStateSaveError):
        store.capture(missing_belief, sequence=20)


def test_legacy_restore_bootstraps_an_attached_attention_owner_without_rewriting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = _store(path)
    legacy = store.capture(_Loop(), sequence=3)
    assert isinstance(legacy, AgentStateSnapshotV7)
    store.save(legacy)
    pretty = json.dumps(json.loads(path.read_text()), indent=2).encode("utf-8")
    path.write_bytes(pretty)
    store.ensure_published(legacy)
    assert path.read_bytes() == pretty

    target_attention = AttentionSystem(_continuity(sequence=8))
    target = _Loop()
    _attach_attention(target, target_attention)
    store.restore_into(target, legacy)
    assert target_attention.snapshot() == AttentionContinuity.bootstrap()
    assert path.read_bytes() == pretty

    v8_path = tmp_path / "agent_state_v8.json"
    v8_store = _store(v8_path)
    v8_snapshot, _source = _capture_v8(v8_store, populated=False)
    v8_target_attention = AttentionSystem(_continuity(sequence=8))
    v8_target = _Loop()
    _attach_attention(v8_target, v8_target_attention)
    v8_store.restore_into(v8_target, v8_snapshot)
    assert v8_target_attention.snapshot() == AttentionContinuity.bootstrap()

    v8_store.save(v8_snapshot)
    v8_pretty = json.dumps(json.loads(v8_path.read_text()), indent=2).encode("utf-8")
    v8_path.write_bytes(v8_pretty)
    v8_mtime = v8_path.stat().st_mtime_ns
    v8_store.ensure_published(v8_snapshot)
    assert v8_path.read_bytes() == v8_pretty
    assert v8_path.stat().st_mtime_ns == v8_mtime


def test_existing_legacy_numeric_version_bytes_are_not_eagerly_normalized(tmp_path: Path) -> None:
    store = _store(tmp_path / "legacy_numeric_version.json")
    snapshot, _source = _capture_v8(store, populated=False)
    payload = store.canonical_bytes(snapshot).replace(
        b'"schema_version":8', b'"schema_version":8.0', 1
    )
    store.path.write_bytes(payload)
    # Retained v8 already accepts this numeric representation. U4 strengthens
    # the new v9 discriminator, not the historical v1-v8 parsing contract.
    assert store.load() == snapshot
    store.ensure_published(snapshot)
    assert store.path.read_bytes() == payload


def test_ensure_published_canonicalizes_a_physically_v0_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy_v0.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "last_event_sequence": 7,
                "emotion": {
                    "valence": 0.1,
                    "arousal": 0.2,
                    "optimal_loss": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )
    store = _store(path)
    migrated = store.load()
    assert isinstance(migrated, AgentStateSnapshotV7)

    store.ensure_published(migrated)

    assert path.read_bytes() == store.canonical_bytes(migrated)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 7


def test_first_v9_capacity_is_derived_from_the_exact_attention_field_bound(
    tmp_path: Path,
) -> None:
    assert AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES == 4_193_496
    assert AGENT_STATE_V8_BASE_MAX_SERIALIZED_BYTES == 109_573_688
    assert AGENT_STATE_V8_SCHEMA_MAX_SERIALIZED_BYTES == 126_350_904
    assert AGENT_STATE_V9_BASE_MAX_SERIALIZED_BYTES == 113_767_203
    assert AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES == 130_544_419
    assert (
        AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES
        < AGENT_STATE_MAX_SERIALIZED_BYTES
    )

    projected_from_v8 = project_agent_state_schema_max_bytes(
        schema_version=9,
        added_field_maxima={
            "attention_state": AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES
        },
        base_schema_version=8,
    )
    projected_v9 = project_agent_state_schema_max_bytes(
        schema_version=9,
        base_schema_version=9,
    )
    assert projected_from_v8 == AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES
    assert projected_v9 == AGENT_STATE_V9_SCHEMA_MAX_SERIALIZED_BYTES

    attention = maximum_attention_continuity_fixture()
    assert (
        len(attention.canonical_bytes())
        == AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES
    )
    base, _owner = _capture_v8(_store(tmp_path / "max-base.json"), populated=False)
    maximum_snapshot = _v9_from_v8(
        base,
        attention,
        sequence=MAX_PERSISTED_EVENT_SEQUENCE,
        saved_at=datetime.max.replace(tzinfo=UTC),
    )
    payload = _store(tmp_path / "maximum.json").canonical_bytes(maximum_snapshot)
    assert len(payload) <= AGENT_STATE_MAX_SERIALIZED_BYTES
    assert json.loads(payload)["attention_state"] == attention.canonical_value()


def test_v8_schema_and_capture_remain_explicitly_legacy(tmp_path: Path) -> None:
    store = _store(tmp_path / "v8.json")
    snapshot, _source = _capture_v8(store, populated=False)
    assert CURRENT_AGENT_STATE_SCHEMA_VERSION == 9
    assert snapshot.schema_version == 8
    assert isinstance(
        AgentStateSnapshotV8.model_validate(snapshot.model_dump()),
        AgentStateSnapshotV8,
    )
    store.save(snapshot)
    loaded = store.load()
    assert isinstance(loaded, AgentStateSnapshotV8)
    assert store.canonical_bytes(loaded) == store.canonical_bytes(snapshot)


def test_deep_legacy_json_is_reported_as_malformed_without_rewriting(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "agent_state.json")
    snapshot, _source = _capture_v8(store, populated=False)
    canonical = store.canonical_bytes(snapshot)
    nested_unknown = b"[" * 1_500 + b"0" + b"]" * 1_500
    payload = canonical[:-1] + b',"legacy_unknown":' + nested_unknown + b"}"
    preflight_attention_json(payload)
    store.path.write_bytes(payload)

    with pytest.raises(AgentStateLoadError, match="snapshot is malformed"):
        store.load()

    assert store.path.read_bytes() == payload


def test_tampered_typed_attention_digest_is_not_repaired_for_persistence(
    tmp_path: Path,
) -> None:
    store, snapshot, _source, _owner, _attention = _captured_v9(tmp_path)
    event = snapshot.attention_state.last_event
    assert event is not None
    object.__setattr__(event, "event_sequence", event.event_sequence - 1)

    with pytest.raises(AgentStateSaveError):
        store.canonical_bytes(snapshot)


def test_attention_prepare_failure_precedes_other_domain_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, snapshot, _source, _source_r13, _source_attention = _captured_v9(tmp_path)
    target_attention = AttentionSystem(_continuity(sequence=8))
    target = _Loop()
    target, target_r13 = _attach_attention(target, target_attention)
    before_value = target.value_system
    before_emotion = target.emotion_engine.state
    before_memory = (target.working_memory.revision, target.working_memory.items)
    before_belief = target.export_belief_state()

    def fail_prepare(
        _port: AttentionStatePort,
        _snapshot: AttentionContinuity,
    ) -> object:
        raise RuntimeError("injected Attention preparation failure")

    monkeypatch.setattr(AttentionStatePort, "prepare_attention_restore", fail_prepare)
    with pytest.raises(AgentStateLoadError, match="could not be prepared"):
        store.restore_into(target, snapshot)

    assert target.value_system is before_value
    assert target.emotion_engine.state == before_emotion
    assert (
        target.working_memory.revision,
        target.working_memory.items,
    ) == before_memory
    assert target.export_belief_state() == before_belief
    assert target_r13.restore_calls == []


class _FailingReadViewLoop(_Loop):
    def _prepare_committed_read_views(self, *_args: object) -> object:
        return object()

    def _publish_committed_read_views(self, _bundle: object) -> None:
        raise RuntimeError("injected post-Attention-publication failure")


def test_failed_read_view_publication_restores_the_exact_prior_attention_bundle(
    tmp_path: Path,
) -> None:
    store, snapshot, _source, _source_r13, _source_attention = _captured_v9(tmp_path)
    prior_attention = AttentionSystem(_continuity(sequence=8))
    prior_bundle = prior_attention._bundle
    target = _FailingReadViewLoop()
    target, target_r13 = _attach_attention(target, prior_attention)
    before_emotion = target.emotion_engine.state
    before_memory = (target.working_memory.revision, target.working_memory.items)
    before_context = target.context_registry.state
    before_belief = target.export_belief_state()
    before_r13 = target_r13.r13_state()

    with pytest.raises(AgentStateLoadError, match="restore failed"):
        store.restore_into(target, snapshot)

    assert prior_attention._bundle is prior_bundle
    assert prior_attention.snapshot() == prior_bundle.snapshot
    assert target.emotion_engine.state == before_emotion
    assert (
        target.working_memory.revision,
        target.working_memory.items,
    ) == before_memory
    assert target.context_registry.state == before_context
    assert target.export_belief_state() == before_belief
    assert target_r13.r13_state() == before_r13
