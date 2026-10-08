"""U5 integration tests for per-chat Attention refresh and read boundaries."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

import suzka.runtime.main_loop as main_loop_module
from suzka.attention.common import (
    ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
    AttentionTargetKind,
    CandidateAvailability,
)
from suzka.attention.contracts import (
    AttentionCandidateProjection,
    AttentionContinuity,
    AttentionEvent,
)
from suzka.attention.system import AttentionRefreshResult, AttentionSystem
from suzka.config import Settings, load_settings
from suzka.memory import DualMemorySystem, MemoryContext
from suzka.models import DummyProvider
from suzka.persona import PromptBuilder
from suzka.persona.attention_prompt import AttentionPromptPayload
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeExecutionError,
)
from suzka.runtime.attention_state_port import AttentionStatePort
from suzka.runtime.main_loop import SuzkaMainLoop
from suzka.runtime.r13_codec import R13StateSnapshot
from suzka.runtime.working_memory import (
    WorkingMemory,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemorySourceKind,
)
from suzka.working_memory_contracts import MAX_ITEM_CAPACITY
from r14_attention_fixtures import coherent_r13_systems


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class _RecordingProvider(DummyProvider):
    response_text = "U5 response"

    def __init__(self, order: list[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.order = order

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.order is not None:
            self.order.append("generate")
        return self.response_text


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "attention_main_loop_episodic",
                    "db2_collection": "attention_main_loop_semantic",
                }
            )
        }
    )


def _runtime(initial_sequence: int = 20) -> AgentRuntime:
    return AgentRuntime(2, initial_sequence=initial_sequence, allow_volatile=True)


def test_main_loop_owns_an_isolated_attention_system_and_concrete_state_port(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    injected = AttentionSystem()
    original = injected.snapshot()
    loop = SuzkaMainLoop(
        settings,
        _RecordingProvider(),
        DualMemorySystem(settings),
        attention_system=injected,
    )

    ports = loop.agent_state_ports
    assert loop._attention_system is not injected
    assert loop._attention_system.snapshot() == original
    assert injected.snapshot() == original
    assert ports.motivation_state_port is loop
    assert ports.goal_state_port is loop
    assert ports.commitment_state_port is loop
    assert type(ports.attention_state_port) is AttentionStatePort
    assert ports.attention_state_port._system is loop._attention_system
    assert not hasattr(loop, "attention_system")
    assert not hasattr(loop, "refresh_attention")
    assert loop.attention_view().revision == 0


def test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    provider = _RecordingProvider(order)
    working_memory = WorkingMemory(item_capacity=8, projection_max_bytes=1_024)
    working_memory.admit(
        WorkingMemorySourceKind.SEMANTIC,
        "current-but-unavailable",
        activation=0.9,
        salience=0.7,
    )
    motivation, goals, commitments = coherent_r13_systems(1)
    settings = _settings(tmp_path)
    loop = SuzkaMainLoop(
        settings,
        provider,
        DualMemorySystem(settings),
        working_memory=working_memory,
        motivation_system=motivation,
        goal_system=goals,
        commitment_system=commitments,
    )
    runtime = _runtime()
    loop.bind_runtime(runtime)
    captured_projections: list[tuple[AttentionCandidateProjection, ...]] = []
    captured_payloads: list[AttentionPromptPayload] = []
    captured_events: list[AttentionEvent] = []

    original_project = main_loop_module.project_attention_candidates

    def project_spy(**kwargs: object) -> tuple[AttentionCandidateProjection, ...]:
        result = original_project(**kwargs)  # type: ignore[arg-type]
        captured_projections.append(result)
        order.append("project")
        return result

    original_refresh = AttentionSystem.refresh

    def refresh_spy(
        owner: AttentionSystem,
        projections: tuple[object, ...],
        event: AttentionEvent,
        *,
        global_emotion: object = None,
    ) -> AttentionRefreshResult:
        order.append("refresh")
        captured_events.append(event)
        return original_refresh(
            owner,
            projections,  # type: ignore[arg-type]
            event,
            global_emotion=global_emotion,  # type: ignore[arg-type]
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
        captured_payloads.append(payload)
        assert loop.attention_view().revision == 0
        order.append("builder")
        return original_build(
            builder,
            user_input,
            emotion_state,  # type: ignore[arg-type]
            working_memory_view,  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        main_loop_module, "project_attention_candidates", project_spy
    )
    monkeypatch.setattr(AttentionSystem, "refresh", refresh_spy)
    monkeypatch.setattr(PromptBuilder, "build", build_spy)
    runtime.start()
    try:
        ordinary = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: loop.chat("ordinary attention turn"),
        ).result(timeout=5)
        assert ordinary.event.processing_sequence == 21
        assert ordinary.event.event_id
        assert ordinary.value is not None
        assert order == ["project", "refresh", "builder", "generate"]

        order.clear()
        debug = runtime.submit(
            AgentEventType.DEBUG_CHAT,
            AgentEventSource.API_CHAT_DEBUG,
            lambda: loop.chat_debug("debug attention turn"),
        ).result(timeout=5)
        assert debug.event.processing_sequence == 22
        assert debug.value is not None
        assert order == ["project", "refresh", "builder", "generate"]
    finally:
        runtime.shutdown()

    assert len(captured_projections) == len(captured_payloads) == 2
    assert tuple(event.event_sequence for event in captured_events) == (21, 22)
    for event, projections in zip(
        captured_events, captured_projections, strict=True
    ):
        assert all(
            projection.event == event for projection in projections
        )
    first = captured_projections[0]
    targets = {projection.target.kind for projection in first}
    assert {
        AttentionTargetKind.WORKING_MEMORY,
        AttentionTargetKind.MOTIVATION,
        AttentionTargetKind.GOAL,
        AttentionTargetKind.COMMITMENT,
    } <= targets
    unavailable = tuple(
        projection
        for projection in first
        if (
            projection.target.kind is AttentionTargetKind.WORKING_MEMORY
            and projection.availability is CandidateAvailability.UNAVAILABLE
        )
    )
    assert len(unavailable) == 1
    assert loop._attention_system.snapshot().revision == 2
    assert loop.attention_view().revision == 0
    assert loop.context_registry.current_context is not None
    assert "ordinary attention turn" in provider.prompts[0]
    assert "debug attention turn" in provider.prompts[1]


def test_real_main_loop_projects_the_complete_4192_candidate_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain_count = ATTENTION_MAX_R13_RECORDS_PER_DOMAIN
    assert domain_count == 32
    settings = _settings(tmp_path)
    provider = _RecordingProvider()
    memory = DualMemorySystem(settings)
    monkeypatch.setattr(
        memory, "retrieve_context", lambda _query: MemoryContext()
    )
    working_memory = WorkingMemory(
        item_capacity=MAX_ITEM_CAPACITY,
        projection_max_bytes=settings.working_memory.projection_max_bytes,
    )
    for index in range(MAX_ITEM_CAPACITY):
        working_memory.admit(
            WorkingMemorySourceKind.SEMANTIC,
            f"full-source-{index:04}",
            activation=0.9,
            salience=0.7,
        )
    motivation, goals, commitments = coherent_r13_systems(domain_count)
    loop = SuzkaMainLoop(
        settings,
        provider,
        memory,
        working_memory=working_memory,
        motivation_system=motivation,
        goal_system=goals,
        commitment_system=commitments,
    )

    class UnavailableCurrentSourceResolver:
        def __call__(self, _item: object) -> WorkingMemoryResolution:
            return WorkingMemoryResolution(
                WorkingMemoryResolutionStatus.UNAVAILABLE
            )

        def resolve_contextual(
            self,
            _item: object,
            _context_registry: object,
            _current_context_id: str,
        ) -> WorkingMemoryResolution:
            return WorkingMemoryResolution(
                WorkingMemoryResolutionStatus.UNAVAILABLE
            )

    loop.working_memory_resolver = (
        UnavailableCurrentSourceResolver()  # type: ignore[assignment]
    )
    initial_sequence = 7 * domain_count + 100
    runtime = _runtime(initial_sequence)
    loop.bind_runtime(runtime)
    runtime.start()
    try:
        result = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: loop.chat("project every current source"),
        ).result(timeout=30)
    finally:
        runtime.shutdown()

    assert result.event.processing_sequence == initial_sequence + 1
    assert provider.prompts
    snapshot = loop._attention_system.snapshot()
    assert len(snapshot.candidates) == MAX_ITEM_CAPACITY + 3 * domain_count
    counts = {
        kind: sum(candidate.target.kind is kind for candidate in snapshot.candidates)
        for kind in AttentionTargetKind
    }
    assert counts == {
        AttentionTargetKind.WORKING_MEMORY: MAX_ITEM_CAPACITY,
        AttentionTargetKind.MOTIVATION: domain_count,
        AttentionTargetKind.GOAL: domain_count,
        AttentionTargetKind.COMMITMENT: domain_count,
    }
    assert all(
        candidate.availability is CandidateAvailability.ELIGIBLE
        for candidate in snapshot.candidates
        if candidate.target.kind is not AttentionTargetKind.WORKING_MEMORY
    )
    wm_candidates = tuple(
        candidate
        for candidate in snapshot.candidates
        if candidate.target.kind is AttentionTargetKind.WORKING_MEMORY
    )
    assert all(
        candidate.availability is CandidateAvailability.UNAVAILABLE
        for candidate in wm_candidates
    )


def test_eventless_legacy_calls_and_out_of_handler_calls_do_not_refresh(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    provider = _RecordingProvider()

    class LegacyBuilder:
        def build(
            self,
            user_input: str,
            emotion_state: object,
            working_memory_view: object,
            context_view: object = None,
            value_view: object = None,
            r13_view: object = None,
        ) -> str:
            return f"legacy:{user_input}"

    loop = SuzkaMainLoop(
        settings,
        provider,
        DualMemorySystem(settings),
        prompt_builder=LegacyBuilder(),  # type: ignore[arg-type]
    )
    loop.chat("unbound legacy turn")
    assert provider.prompts == ["legacy:unbound legacy turn"]
    assert loop._attention_system.snapshot().revision == 0

    bound = SuzkaMainLoop(settings, _RecordingProvider(), DualMemorySystem(settings))
    runtime = _runtime()
    bound.bind_runtime(runtime)
    with pytest.raises(RuntimeError, match="active runtime event"):
        bound.chat("outside runtime worker")
    assert bound._attention_system.snapshot().revision == 0


def test_payload_builder_must_be_exact_and_failure_keeps_committed_attention_view(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    provider = _RecordingProvider()

    class UnsupportedBuilder(PromptBuilder):
        pass

    loop = SuzkaMainLoop(
        settings,
        provider,
        DualMemorySystem(settings),
        prompt_builder=UnsupportedBuilder(),
    )
    runtime = _runtime()
    loop.bind_runtime(runtime)
    runtime.start()
    try:
        future = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: loop.chat("must not generate"),
        )
        with pytest.raises(AgentRuntimeExecutionError):
            future.result(timeout=5)
    finally:
        runtime.shutdown()

    assert provider.prompts == []
    assert loop._attention_system.snapshot().revision == 1
    assert loop.attention_view().revision == 0


def test_read_view_preparation_preserves_committed_metadata_and_targets_restore_root(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    loop = SuzkaMainLoop(settings, _RecordingProvider(), DualMemorySystem(settings))
    event = AttentionEvent(
        "event:staged-attention",
        1,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    refresh = loop._attention_system.refresh((), event)
    assert refresh.view.revision == 1

    r13_state = R13StateSnapshot.empty()
    prepared_legacy = loop._prepare_committed_read_views(
        loop._value_system.snapshot(),
        loop._belief_system.snapshot(),
        r13_state,
    )
    assert prepared_legacy.attention_view.revision == 0

    retained = loop._prepare_committed_attention_read_views(
        loop._value_system.snapshot(),
        loop._belief_system.snapshot(),
        r13_state,
        refresh.snapshot,
        retain_selection=True,
    )
    assert retained.attention_view.revision == 1
    assert retained.attention_view.event == event

    restored = loop._prepare_committed_attention_read_views(
        loop._value_system.snapshot(),
        loop._belief_system.snapshot(),
        r13_state,
        AttentionContinuity.bootstrap(),
        retain_selection=False,
    )
    assert restored.attention_view.revision == 0
    assert restored.attention_view.event is None
