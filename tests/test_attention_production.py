"""Production Attention refresh, prompt selection, and commit visibility tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from suzka.attention.common import (
    ATTENTION_PROMPT_BUDGET_BYTES,
    AttentionTargetKind,
    SourceDigestKind,
)
from suzka.attention.policy import AttentionPromptReason
from suzka.config import Settings
from suzka.identity import ValueConflictDefinition
from suzka.memory import DualMemorySystem, MemoryRecordType
from suzka.models import DummyProvider
from suzka.motivation import (
    R13_MAX_RECORDS_PER_DOMAIN,
)
from suzka.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeDurabilityError,
    AgentRuntimeExecutionError,
    AgentRuntimeStatus,
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshotV8,
    AgentStateSnapshotV9,
    AgentStateStore,
    CompatibleAgentStateSnapshot,
    ContextRegistry,
    ContextType,
    SuzkaMainLoop,
)
from test_fastapi_backend import _client, _settings
from r14_attention_fixtures import coherent_r13_systems


SOURCE_EPISODE_ID = "episode-attention-production-source"
SOURCE_QUERY = "retrieve the cedar source fact"
SOURCE_BODY = "Verified source fact: preserve the cedar token."


class _ProbingProvider(DummyProvider):
    def __init__(self, *, fail_generation: bool = False) -> None:
        self.main_loop: SuzkaMainLoop | None = None
        self.prompts: list[str] = []
        self.views: list[object] = []
        self.fail_generation = fail_generation

    def generate(self, prompt: str) -> str:
        assert self.main_loop is not None
        self.prompts.append(prompt)
        self.views.append(self.main_loop.attention_view())
        if self.fail_generation:
            raise RuntimeError("injected model failure")
        return "Attention production response."


def _seed_retained_v8(settings: Settings) -> AgentStateSnapshotV8:
    """Seed real source authorities, then retain their compatible v8 root."""

    memory = DualMemorySystem(settings)
    memory.publish_coordinated_episodic(
        SOURCE_EPISODE_ID,
        SOURCE_QUERY,
        SOURCE_BODY,
        loss=0.1,
        emotion_valence=0.2,
        emotion_arousal=0.3,
        record_type=MemoryRecordType.EPISODIC_LOG,
        created_at=datetime.now(UTC).isoformat(),
        coordination_schema=3,
        context_id="conversation.default",
        source_channel="chat",
    )

    context_registry = ContextRegistry()
    context_registry.create(
        "conversation.default",
        ContextType.CONVERSATION,
        "chat",
    )
    context_registry.set_current("conversation.default")
    motivation, goal, commitment = coherent_r13_systems(
        R13_MAX_RECORDS_PER_DOMAIN
    )
    loop = SuzkaMainLoop(
        settings,
        DummyProvider(),
        memory,
        context_registry=context_registry,
        motivation_system=motivation,
        goal_system=goal,
        commitment_system=commitment,
    )

    store = AgentStateStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
    )
    store.configure_value_contract(
        tuple(seed.to_declaration() for seed in settings.values.seeds),
        tuple(
            ValueConflictDefinition(
                left_value_id=conflict.left_value_id,
                right_value_id=conflict.right_value_id,
            )
            for conflict in settings.values.conflicts
        ),
    )
    captured = store.capture(loop, sequence=200)
    assert isinstance(captured, AgentStateSnapshotV9)
    legacy_value = captured.model_dump(mode="python")
    legacy_value.pop("attention_state")
    legacy_value["schema_version"] = 8
    retained = AgentStateSnapshotV8.model_validate(legacy_value)
    store.save(retained)
    return retained


def _assert_full_source_coverage(snapshot: AgentStateSnapshotV9) -> None:
    attention = snapshot.attention_state
    current_r13 = snapshot.r13_state
    expected = {
        AttentionTargetKind.WORKING_MEMORY: len(snapshot.working_memory.items),
        AttentionTargetKind.MOTIVATION: sum(
            record.lifecycle.value == "active"
            for record in current_r13.motivation.restore().records
        ),
        AttentionTargetKind.GOAL: sum(
            record.lifecycle.value == "adopted"
            for record in current_r13.goal.restore().records
        ),
        AttentionTargetKind.COMMITMENT: sum(
            record.lifecycle.value == "active"
            for record in current_r13.commitment.restore().records
        ),
    }
    actual = {
        kind: sum(candidate.target.kind is kind for candidate in attention.candidates)
        for kind in AttentionTargetKind
    }
    assert actual == expected


def test_production_chat_refreshes_and_selects_from_current_sources_before_prompt(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    retained = _seed_retained_v8(settings)
    retained_bytes = settings.agent_state.path.read_bytes()
    provider = _ProbingProvider()

    with _client(tmp_path, settings=settings, provider=provider) as client:
        loop: SuzkaMainLoop = client.app.state.main_loop
        provider.main_loop = loop
        store: AgentStateStore = client.app.state.agent_state_store
        assert store.load() == retained
        assert settings.agent_state.path.read_bytes() == retained_bytes
        before_first_generation = loop.attention_view()
        assert before_first_generation.event is None
        assert before_first_generation.competition is None
        assert before_first_generation.prompt is None

        first_response = client.post(
            "/api/chat",
            json={"message": SOURCE_QUERY, "attachments": []},
        )
        assert first_response.status_code == 200
        assert provider.views[0] == before_first_generation

        first = store.load()
        assert isinstance(first, AgentStateSnapshotV9)
        first_event_record = client.app.state.event_journal.records[-1]
        assert first_event_record.event_id is not None
        assert first_event_record.processing_sequence == 201
        attention = first.attention_state
        assert attention.revision == 1
        assert attention.last_event is not None
        assert attention.last_event.event_id == first_event_record.event_id
        assert (
            attention.last_event.event_sequence
            == first_event_record.processing_sequence
        )
        assert first.context_state.current_context_id == "conversation.default"
        _assert_full_source_coverage(first)

        committed_view = loop.attention_view()
        assert committed_view.event == attention.last_event
        assert committed_view.competition is not None
        selection = committed_view.prompt
        assert selection is not None
        assert selection.total_bytes <= ATTENTION_PROMPT_BUDGET_BYTES
        assert len(attention.candidates) == 1 + 3 * R13_MAX_RECORDS_PER_DOMAIN
        assert len(selection.decisions) == len(attention.candidates)
        assert len(selection.included_candidate_ids) <= 16
        assert len(selection.included_candidate_ids) < 3 * R13_MAX_RECORDS_PER_DOMAIN
        included = set(selection.included_candidate_ids)
        assert included == {
            decision.candidate_id
            for decision in selection.decisions
            if decision.reason is AttentionPromptReason.INCLUDED
        }

        prompt = provider.prompts[0]
        assert SOURCE_BODY in prompt
        assert SOURCE_EPISODE_ID not in prompt
        wm_candidate = next(
            candidate
            for candidate in attention.candidates
            if candidate.target.kind is AttentionTargetKind.WORKING_MEMORY
        )
        assert wm_candidate.candidate_id in included
        assert wm_candidate.target.reference not in prompt
        assert wm_candidate.source.digest_kind is SourceDigestKind.ATTENTION_PROJECTION
        selected_r13_candidate = next(
            candidate
            for candidate in attention.candidates
            if candidate.target.kind is not AttentionTargetKind.WORKING_MEMORY
            and candidate.candidate_id in included
        )
        assert selected_r13_candidate.target.reference in prompt
        assert "private subject 000" not in prompt
        rendered_witness = next(
            decision
            for decision in selection.decisions
            if decision.candidate_id == wm_candidate.candidate_id
        )
        assert rendered_witness.rendered_bytes is not None
        assert rendered_witness.rendered_digest is not None
        assert rendered_witness.rendered_digest != wm_candidate.source.digest

        stored_source = client.app.state.memory_system.get_committed_episodic(
            SOURCE_EPISODE_ID
        )
        assert stored_source is not None
        assert stored_source.record.context_id == first.context_state.current_context_id
        assert {candidate.target.kind for candidate in attention.candidates} >= {
            AttentionTargetKind.MOTIVATION,
            AttentionTargetKind.GOAL,
            AttentionTargetKind.COMMITMENT,
        }
        assert first.r13_state == retained.r13_state

        before_second_generation = loop.attention_view()
        second_response = client.post(
            "/api/chat",
            json={"message": "repeat the cedar source retrieval", "attachments": []},
        )
        assert second_response.status_code == 200
        assert provider.views[1] == before_second_generation

        second = store.load()
        assert isinstance(second, AgentStateSnapshotV9)
        second_event_record = client.app.state.event_journal.records[-1]
        assert second_event_record.event_id is not None
        assert second_event_record.processing_sequence == 202
        assert second.attention_state.revision == 2
        assert second.attention_state.last_event is not None
        assert second.attention_state.last_event.event_id == second_event_record.event_id
        assert second.attention_state.last_event.event_sequence == 202
        assert loop.attention_view().event == second.attention_state.last_event
        assert second.r13_state == retained.r13_state


def test_committed_attention_survives_nonchat_commit_then_failed_next_chat(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _seed_retained_v8(settings)
    provider = _ProbingProvider()

    with _client(tmp_path, settings=settings, provider=provider) as client:
        loop: SuzkaMainLoop = client.app.state.main_loop
        provider.main_loop = loop
        response = client.post(
            "/api/chat",
            json={"message": SOURCE_QUERY, "attachments": []},
        )
        assert response.status_code == 200
        first_committed = client.app.state.agent_state_store.load()
        assert isinstance(first_committed, AgentStateSnapshotV9)
        before_view = loop.attention_view()
        assert before_view.event == first_committed.attention_state.last_event

        tick = client.app.state.agent_runtime.submit(
            AgentEventType.EMOTION_TICK,
            AgentEventSource.RUNTIME_EMOTION_TIMER,
            loop.emotion_tick,
        ).result(timeout=60)
        assert tick.value is None
        after_tick = client.app.state.agent_state_store.load()
        assert isinstance(after_tick, AgentStateSnapshotV9)
        assert after_tick.last_processed_event_sequence == 202
        assert after_tick.attention_state == first_committed.attention_state
        assert loop.attention_view() == before_view

        before_bytes = settings.agent_state.path.read_bytes()
        before_wal = client.app.state.state_wal.inspect()
        provider.fail_generation = True

        with pytest.raises(AgentRuntimeExecutionError):
            client.app.state.agent_runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                lambda: loop.chat("fail after a committed Attention turn"),
            ).result(timeout=60)

        assert provider.views[1] == before_view
        assert loop.attention_view() == before_view
        attention_port = loop.agent_state_ports.attention_state_port
        assert attention_port.export_attention_state() == after_tick.attention_state
        assert client.app.state.agent_state_store.load() == after_tick
        assert settings.agent_state.path.read_bytes() == before_bytes
        assert client.app.state.state_wal.inspect() == before_wal
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING


def test_read_view_preflight_failure_precedes_internal_journal_and_wal_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    retained = _seed_retained_v8(settings)
    provider = _ProbingProvider()

    with _client(tmp_path, settings=settings, provider=provider) as client:
        loop: SuzkaMainLoop = client.app.state.main_loop
        provider.main_loop = loop
        before_view = loop.attention_view()
        before_bytes = settings.agent_state.path.read_bytes()
        before_wal = client.app.state.state_wal.inspect()

        def reject_read_view(*_args: object, **_kwargs: object) -> object:
            raise ValueError("injected committed Attention view preflight failure")

        monkeypatch.setattr(
            loop,
            "_prepare_committed_attention_read_views",
            reject_read_view,
        )
        with pytest.raises(AgentRuntimeDurabilityError) as raised:
            client.app.state.agent_runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                lambda: loop.chat(SOURCE_QUERY),
            ).result(timeout=60)

        failed_event_id = raised.value.event.event_id
        assert raised.value.phase.value == "internal_commit"
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert provider.views == [before_view]
        assert loop.attention_view() == before_view
        assert client.app.state.agent_state_store.load() == retained
        assert settings.agent_state.path.read_bytes() == before_bytes
        assert client.app.state.state_wal.inspect() == before_wal
        assert not any(
            record.event_id == failed_event_id
            and record.lifecycle is not None
            and record.lifecycle.value == "prepared"
            for record in client.app.state.event_journal.records
        )


def test_published_snapshot_failure_keeps_prior_view_and_restarts_from_v9_focus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _seed_retained_v8(settings)
    provider = _ProbingProvider()

    with _client(tmp_path, settings=settings, provider=provider) as client:
        loop: SuzkaMainLoop = client.app.state.main_loop
        provider.main_loop = loop
        first_response = client.post(
            "/api/chat",
            json={"message": SOURCE_QUERY, "attachments": []},
        )
        assert first_response.status_code == 200
        store: AgentStateStore = client.app.state.agent_state_store
        prior_snapshot = store.load()
        assert isinstance(prior_snapshot, AgentStateSnapshotV9)
        prior_view = loop.attention_view()
        prior_bytes = settings.agent_state.path.read_bytes()
        save = store.save

        def publish_then_fail(snapshot: CompatibleAgentStateSnapshot) -> None:
            save(snapshot)
            raise AgentStateSaveError(
                AgentStateSaveStage.PARENT_FSYNC,
                published=True,
            )

        monkeypatch.setattr(store, "save", publish_then_fail)
        with pytest.raises(AgentRuntimeDurabilityError) as raised:
            client.app.state.agent_runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                lambda: loop.chat("publish a candidate then fail its save"),
            ).result(timeout=60)

        assert raised.value.phase.value == "internal_commit"
        assert raised.value.published is True
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert provider.views[1] == prior_view
        assert loop.attention_view() == prior_view
        assert settings.agent_state.path.read_bytes() != prior_bytes
        published = store.load()
        assert isinstance(published, AgentStateSnapshotV9)
        assert published.last_processed_event_sequence == 202
        assert published.attention_state.revision == prior_snapshot.attention_state.revision + 1
        assert published.attention_state.last_event is not None
        assert published.attention_state.last_event.event_sequence == 202

    restart_provider = _ProbingProvider()
    with _client(tmp_path, settings=settings, provider=restart_provider) as restarted:
        loop: SuzkaMainLoop = restarted.app.state.main_loop
        restart_provider.main_loop = loop
        restored = restarted.app.state.agent_state_store.load()
        assert isinstance(restored, AgentStateSnapshotV9)
        assert restored.attention_state == published.attention_state
        restored_view = loop.attention_view()
        assert restored_view.event == restored.attention_state.last_event
        assert restored_view.competition is None
        assert restored_view.prompt is None
        assert tuple(
            target.candidate_id for target in restored_view.focused_targets
        ) == restored.attention_state.focused_ids
