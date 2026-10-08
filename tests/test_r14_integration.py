"""Bounded cross-boundary regressions for the R14 U6 integration gate."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import suzka.attention.system as attention_system_module
import suzka.metacognition as metacognition
import suzka.metacognition.assessment as metacognition_assessment
import suzka.metacognition.evidence as metacognition_evidence
from suzka.attention.common import AttentionSourceKind, AttentionTargetKind
from suzka.attention.contracts import AttentionContinuity, AttentionEvent
from suzka.attention.system import AttentionReplayConflict, AttentionSystem
from suzka.belief import BeliefSystemSnapshot
from suzka.config import Settings
from suzka.identity import ValueConflictDefinition
from suzka.memory import DualMemorySystem
from suzka.metacognition.contracts import EpistemicBoundary, EvidenceCondition
from suzka.motivation import (
    CommitmentSystemSnapshot,
    GoalSystemSnapshot,
    MotivationSystemSnapshot,
)
from suzka.persona import AttentionPromptPayload, PromptBuilder
from suzka.persona.conscious_agent import ConsciousAgent
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeExecutionError,
)
from suzka.runtime.agent_state import AgentStateSnapshotV9, AgentStateStore
from suzka.runtime.context import ContextRegistry, ContextType
from suzka.runtime.main_loop import SuzkaMainLoop
from test_attention_main_loop import _RecordingProvider, _runtime
from test_attention_production import (
    _ProbingProvider,
    _client,
    _seed_retained_v8,
    _settings,
)


def _source_roots(loop: SuzkaMainLoop) -> tuple[object, ...]:
    """Capture immutable roots on both sides of the test-only U3 call."""

    return (
        loop.context_registry.state,
        loop.export_belief_state(),
        loop.export_motivation_state(),
        loop.export_goal_state(),
        loop.export_commitment_state(),
        loop.agent_state_ports.attention_state_port.export_attention_state(),
        loop.working_memory.revision,
        loop.working_memory.items,
        loop.emotion_engine.state,
    )


def _focus_records(
    attention: AttentionContinuity,
    motivation: MotivationSystemSnapshot,
    goals: GoalSystemSnapshot,
    commitments: CommitmentSystemSnapshot,
) -> tuple[tuple[object, ...], object]:
    focused_targets = {
        (candidate.target.kind, candidate.target.reference)
        for candidate in attention.candidates
        if candidate.candidate_id in attention.focused_ids
    }
    source_records = tuple(
        (kind, getattr(record, identifier), record)
        for kind, identifier, records in (
            (AttentionTargetKind.MOTIVATION, "motivation_id", motivation.records),
            (AttentionTargetKind.GOAL, "goal_id", goals.records),
            (AttentionTargetKind.COMMITMENT, "commitment_id", commitments.records),
        )
        for record in records
    )
    focus_records = tuple(
        record
        for kind, reference, record in source_records
        if (kind, reference) in focused_targets
    )
    unfocused_record = next(
        record
        for kind, reference, record in source_records
        if (kind, reference) not in focused_targets
    )
    return focus_records, unfocused_record


def _commit_production_v9(
    tmp_path: Path,
) -> tuple[Settings, AgentStateSnapshotV9, bytes]:
    settings = _settings(tmp_path)
    _seed_retained_v8(settings)
    provider = _ProbingProvider()

    with _client(tmp_path, settings=settings, provider=provider) as client:
        loop: SuzkaMainLoop = client.app.state.main_loop
        provider.main_loop = loop
        client.app.state.agent_runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: loop.chat("commit one ordinary v9 chat turn"),
        ).result(timeout=60)
        snapshot = client.app.state.agent_state_store.load()
        assert isinstance(snapshot, AgentStateSnapshotV9)
        payload = settings.agent_state.path.read_bytes()
        assert payload == client.app.state.agent_state_store.canonical_bytes(snapshot)

    return settings, snapshot, payload


def _agent_state_store(
    settings: Settings,
    *,
    clock: datetime,
) -> AgentStateStore:
    conflicts = tuple(
        ValueConflictDefinition(
            left_value_id=item.left_value_id,
            right_value_id=item.right_value_id,
        )
        for item in settings.values.conflicts
    )
    return AgentStateStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
        value_seeds=tuple(seed.to_declaration() for seed in settings.values.seeds),
        value_conflicts=conflicts,
        clock=lambda: clock,
    )


def test_test_harness_metacognition_observes_the_exact_live_r14_turn_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _seed_retained_v8(settings)
    provider = _ProbingProvider()
    observed_values: list[tuple[object, object]] = []
    built_prompts: list[str] = []
    prompt_payloads: list[AttentionPromptPayload] = []

    with _client(tmp_path, settings=settings, provider=provider) as client:
        loop: SuzkaMainLoop = client.app.state.main_loop
        provider.main_loop = loop
        runtime: AgentRuntime = client.app.state.agent_runtime
        initial_committed_view = loop.attention_view()
        original_refresh = loop._refresh_attention_prompt

        def refresh_with_test_only_metacognition(
            event: AgentEvent,
            current_context: object,
            emotion_state: object,
            working_memory_view: object,
            working_memory: object,
        ) -> AttentionPromptPayload:
            active_event = runtime.current_event()
            assert active_event is event
            assert type(event.processing_sequence) is int
            observed_event = AttentionEvent(
                event.event_id,
                event.processing_sequence,
                event.requested_at,
            )

            # U5's existing producer boundary rejects a copied/forged R09 frame
            # before it can refresh Attention or mutate any source authority.
            before_context_rejection = _source_roots(loop)
            forged_context = replace(
                current_context,
                source_session_id="caller-forged-u6-session",
            )
            with pytest.raises(RuntimeError, match="current R09 authority"):
                original_refresh(
                    event,
                    forged_context,
                    emotion_state,
                    working_memory_view,
                    working_memory,
                )
            assert _source_roots(loop) == before_context_rejection

            payload = original_refresh(
                event,
                current_context,
                emotion_state,
                working_memory_view,
                working_memory,
            )
            assert runtime.current_event() is event
            assert loop.context_registry.state.current_context_id == (
                current_context.context_id
            )
            assert loop.context_registry.current_context == current_context

            # The values below are the exact current roots consumed by the
            # production R14 producer, captured while its serialized handler is
            # still active. U3 is called only by this test harness.
            context_state = loop.context_registry.state
            belief_snapshot: BeliefSystemSnapshot = loop.export_belief_state()
            motivation_snapshot = loop.export_motivation_state()
            goal_snapshot = loop.export_goal_state()
            commitment_snapshot = loop.export_commitment_state()
            attention_snapshot = loop._attention_system.snapshot()
            working_memory_items = working_memory.items
            working_memory_revision = working_memory.revision
            assert attention_snapshot.last_event == observed_event
            assert working_memory_view.revision == working_memory_revision
            assert working_memory.items == working_memory_items
            assert emotion_state == loop.emotion_engine.state
            focus_records, unfocused_record = _focus_records(
                attention_snapshot,
                motivation_snapshot,
                goal_snapshot,
                commitment_snapshot,
            )
            assert focus_records, "exercise a positive focused-R13 observation"
            focused_r13_targets = {
                (candidate.target.kind, candidate.target.reference)
                for candidate in attention_snapshot.candidates
                if candidate.candidate_id in attention_snapshot.focused_ids
                and candidate.target.kind is not AttentionTargetKind.WORKING_MEMORY
            }
            assert len(focus_records) == len(focused_r13_targets)
            source_roots_before_observation = _source_roots(loop)
            committed_view_before_observation = loop.attention_view()
            payload_fields_before_observation = (
                payload.event,
                payload.selection_digest,
                payload.included_candidate_ids,
                payload.witnessed_bytes,
                payload.rendered_bytes,
                payload.rendered_text,
                payload.rendered_digest,
            )
            assert committed_view_before_observation == initial_committed_view
            assert provider.prompts == []

            observation = metacognition_evidence.observe_metacognition(
                observed_event,
                attention_snapshot,
                working_memory_items=working_memory_items,
                working_memory_revision=working_memory_revision,
                working_memory_view=working_memory_view,
                emotion_state=emotion_state,
                focus_records=focus_records,
                belief_records=belief_snapshot.records,
                current_context_id=context_state.current_context_id,
            )
            assessment = metacognition_assessment.assess_metacognition(observation)

            assert observation.event == observed_event
            assert observation.focus_witness.event == observed_event
            assert observation.focus_witness.attention_revision == (
                attention_snapshot.revision
            )
            assert observation.focus_witness.attention_state_digest == (
                attention_snapshot.state_digest
            )
            assert assessment.event == observed_event
            assert assessment.epistemic_boundary is EpistemicBoundary.UNCERTAIN
            assert observation.belief_coverage_units is None
            assert observation.belief_confidence_ceiling_units is None
            assert assessment.evidence_sufficiency is None
            assert assessment.confidence is None
            assert not any(
                witness.source_kind is AttentionSourceKind.BELIEF
                and witness.condition is EvidenceCondition.SUPPORTING
                for witness in observation.evidence
            )
            assert all(
                witness.confidence_ceiling is None
                for witness in observation.evidence
                if witness.source_kind is not AttentionSourceKind.BELIEF
            )

            # Same sequence/identity with a different timestamp, then a valid
            # current R13 record that is not in Attention's focused set, are
            # rejected as mismatched witnesses without changing any root.
            mismatched_event = AttentionEvent(
                observed_event.event_id,
                observed_event.event_sequence,
                observed_event.occurred_at + timedelta(microseconds=1),
            )
            with pytest.raises(ValueError, match="exact same event"):
                metacognition_evidence.observe_metacognition(
                    mismatched_event,
                    attention_snapshot,
                    working_memory_items=working_memory_items,
                    working_memory_revision=working_memory_revision,
                    working_memory_view=working_memory_view,
                    emotion_state=emotion_state,
                    focus_records=focus_records,
                    belief_records=belief_snapshot.records,
                    current_context_id=context_state.current_context_id,
                )
            with pytest.raises(ValueError, match="non-focused R13"):
                metacognition_evidence.observe_metacognition(
                    observed_event,
                    attention_snapshot,
                    working_memory_items=working_memory_items,
                    working_memory_revision=working_memory_revision,
                    working_memory_view=working_memory_view,
                    emotion_state=emotion_state,
                    focus_records=(*focus_records, unfocused_record),
                    belief_records=belief_snapshot.records,
                    current_context_id=context_state.current_context_id,
                )

            assert _source_roots(loop) == source_roots_before_observation
            assert loop.attention_view() == committed_view_before_observation
            assert payload_fields_before_observation == (
                payload.event,
                payload.selection_digest,
                payload.included_candidate_ids,
                payload.witnessed_bytes,
                payload.rendered_bytes,
                payload.rendered_text,
                payload.rendered_digest,
            )
            assert provider.prompts == []
            observed_values.append((observation, assessment))
            prompt_payloads.append(payload)
            return payload

        monkeypatch.setattr(
            loop,
            "_refresh_attention_prompt",
            refresh_with_test_only_metacognition,
        )
        original_build = PromptBuilder.build

        def build_spy(
            builder: PromptBuilder,
            user_input: str,
            emotion_state: object,
            working_memory_view: object,
            **kwargs: object,
        ) -> str:
            payload = kwargs.get("attention_payload")
            assert type(payload) is AttentionPromptPayload
            assert prompt_payloads == [payload]
            assert "metacognition" not in kwargs
            assert "assessment" not in kwargs
            result = original_build(
                builder,
                user_input,
                emotion_state,  # type: ignore[arg-type]
                working_memory_view,  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
            assert "metacognition" not in result.lower()
            built_prompts.append(result)
            return result

        monkeypatch.setattr(PromptBuilder, "build", build_spy)
        outcome = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: loop.chat("retrieve the cedar source fact"),
        ).result(timeout=60)

        assert outcome.event.processing_sequence is not None
        assert observed_values
        observation, assessment = observed_values[0]
        expected_event = AttentionEvent(
            outcome.event.event_id,
            outcome.event.processing_sequence,
            outcome.event.requested_at,
        )
        assert observation.event == expected_event
        assert assessment.event == expected_event
        assert len(provider.prompts) == 1
        assert built_prompts == provider.prompts
        assert "metacognition" not in provider.prompts[0].lower()
        assert provider.views == [initial_committed_view]
        assert loop.attention_view().event == expected_event

        persisted = client.app.state.agent_state_store.load()
        assert isinstance(persisted, AgentStateSnapshotV9)
        serialized = settings.agent_state.path.read_bytes()
        assert serialized == client.app.state.agent_state_store.canonical_bytes(
            persisted
        )
        assert persisted.attention_state.last_event == expected_event
        assert loop.agent_state_ports.attention_state_port.export_attention_state() == (
            persisted.attention_state
        )
        assert not {
            name
            for name in persisted.model_dump(mode="python")
            if "metacogn" in name.lower() or "assessment" in name.lower()
        }
        assert b"metacogn" not in serialized.lower()


def test_v9_restart_recovers_without_attention_or_cognition_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, committed, committed_bytes = _commit_production_v9(tmp_path)
    restart_provider = _ProbingProvider()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail(
            "v9 restart attempted policy, cognition, prompt, model, or retrieval"
        )

    # These sentinels are installed before the actual application startup path.
    # Any forbidden recovery/restart replay fails this test immediately.
    monkeypatch.setattr(AttentionSystem, "refresh", forbidden)
    monkeypatch.setattr(attention_system_module, "compete_attention", forbidden)
    monkeypatch.setattr(
        attention_system_module, "select_attention_prompt", forbidden
    )
    monkeypatch.setattr(
        metacognition_evidence, "observe_metacognition", forbidden
    )
    monkeypatch.setattr(
        metacognition_assessment, "assess_metacognition", forbidden
    )
    monkeypatch.setattr(metacognition, "observe_metacognition", forbidden)
    monkeypatch.setattr(metacognition, "assess_metacognition", forbidden)
    monkeypatch.setattr(PromptBuilder, "build", forbidden)
    monkeypatch.setattr(ConsciousAgent, "generate", forbidden)
    monkeypatch.setattr(DualMemorySystem, "retrieve_context", forbidden)

    with _client(
        tmp_path,
        settings=settings,
        provider=restart_provider,
    ) as restarted:
        loop: SuzkaMainLoop = restarted.app.state.main_loop
        restored = restarted.app.state.agent_state_store.load()
        assert restored == committed
        assert settings.agent_state.path.read_bytes() == committed_bytes
        assert loop.attention_view().event == committed.attention_state.last_event
        assert loop.attention_view().competition is None
        assert loop.attention_view().prompt is None
        assert tuple(
            target.candidate_id for target in loop.attention_view().focused_targets
        ) == committed.attention_state.focused_ids
        assert restart_provider.prompts == []


def test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    historical_time = datetime.max.replace(tzinfo=UTC)
    historical_event = AttentionEvent(
        "u6:historical-attention-root",
        20,
        historical_time,
    )
    source_attention = AttentionSystem()
    source_attention.refresh((), historical_event)
    serialized_root = source_attention.snapshot().canonical_bytes()
    restored_root = AttentionContinuity.from_canonical_value(
        source_attention.snapshot().canonical_value()
    )
    context_registry = ContextRegistry()
    context_registry.create(
        "conversation.default",
        ContextType.CONVERSATION,
        "chat",
    )
    context_registry.set_current("conversation.default")
    source_loop = SuzkaMainLoop(
        settings,
        _RecordingProvider(),
        DualMemorySystem(settings),
        context_registry=context_registry,
        attention_system=AttentionSystem(restored_root),
    )
    store = _agent_state_store(
        settings,
        clock=historical_time,
    )
    captured = store.capture(source_loop, sequence=20)
    assert isinstance(captured, AgentStateSnapshotV9)
    store.save(captured)
    persisted_bytes = settings.agent_state.path.read_bytes()

    # Restore a fresh MainLoop from the actual serialized v9 snapshot, then bind
    # a new process-local runtime watermark at the persisted sequence.
    restored_loop = SuzkaMainLoop(
        settings,
        _RecordingProvider(),
        DualMemorySystem(settings),
        context_registry=ContextRegistry(),
    )
    loaded = store.load()
    assert isinstance(loaded, AgentStateSnapshotV9)
    store.restore_into(restored_loop, loaded)
    prior_view = restored_loop.attention_view()
    prior_roots = _source_roots(restored_loop)
    restored_attention = (
        restored_loop.agent_state_ports.attention_state_port.export_attention_state()
    )
    assert restored_attention.canonical_bytes() == serialized_root

    runtime = _runtime(loaded.last_processed_event_sequence)
    restored_loop.bind_runtime(runtime)
    attempted_events: list[AgentEvent] = []

    def attempt_restored_refresh() -> AttentionPromptPayload:
        event = runtime.current_event()
        assert event is not None
        attempted_events.append(event)
        current_context = restored_loop.context_registry.current_context
        assert current_context is not None
        working_memory_view = restored_loop.working_memory.select_contextual(
            restored_loop.working_memory_resolver,
            restored_loop.context_registry,
            current_context.context_id,
        )
        return restored_loop._refresh_attention_prompt(
            event,
            current_context,
            restored_loop.emotion_engine.state,
            working_memory_view,
            restored_loop.working_memory,
        )

    runtime.start()
    try:
        with pytest.raises(AgentRuntimeExecutionError) as rejected:
            runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                attempt_restored_refresh,
            ).result(timeout=10)
    finally:
        runtime.shutdown()

    assert rejected.value.event.processing_sequence == (
        loaded.last_processed_event_sequence + 1
    )
    assert isinstance(rejected.value.__cause__, AttentionReplayConflict)
    assert len(attempted_events) == 1
    assert attempted_events[0].requested_at < historical_event.occurred_at
    assert rejected.value.event.requested_at < historical_event.occurred_at
    assert store.load() == loaded
    assert settings.agent_state.path.read_bytes() == persisted_bytes
    restored_attention = (
        restored_loop.agent_state_ports.attention_state_port.export_attention_state()
    )
    assert restored_attention.canonical_bytes() == serialized_root
    assert restored_loop.attention_view() == prior_view
    assert _source_roots(restored_loop) == prior_roots
    assert restored_loop.provider.prompts == []
