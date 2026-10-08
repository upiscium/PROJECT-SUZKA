"""Regression coverage for request timestamps following runtime admission order."""

from collections.abc import Callable
from concurrent.futures import Future
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Condition, Event, Lock, Thread, current_thread

import pytest

import suzka.runtime.agent_runtime as agent_runtime_module
from suzka.attention.contracts import AttentionEvent
from suzka.attention.system import AttentionRefreshResult, AttentionSystem
from suzka.belief import (
    AdmissionReason,
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefProposition,
    BeliefRecord,
    BeliefSubjectAdmission,
)
from suzka.memory import DualMemorySystem, MemoryContext
from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.goal import (
    GoalAdmissionReason,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    GoalSubjectAdmission,
    goal_id_for_target,
    goal_proposal_digest,
)
from suzka.motivation.goal_system import (
    GoalMutationEvidence,
    GoalSystemEventOperation,
)
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeStatus,
)
from suzka.runtime.main_loop import SuzkaMainLoop
from test_attention_main_loop import _RecordingProvider, _settings


_TIMEOUT = 10
_FIRST_REQUEST_AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("callback_phase", ["guard", "checkpoint"])
def test_reentrant_admission_is_refused_before_it_can_invert_fifo_time(
    monkeypatch: pytest.MonkeyPatch, callback_phase: str
) -> None:
    _install_clock(monkeypatch)
    accepted: list[AgentEvent] = []
    blocked: list[AgentEvent] = []
    guard_calls: list[AgentEvent] = []
    nested_handler_calls: list[str] = []
    attempted = False

    def nested_submission() -> None:
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(AgentRuntimeAdmissionBlocked) as rejection:
                runtime.submit(
                    AgentEventType.CHAT,
                    AgentEventSource.API_CHAT,
                    lambda: nested_handler_calls.append("nested"),
                )
            blocked.append(rejection.value.event)

    def guard(event: AgentEvent) -> None:
        guard_calls.append(event)
        if callback_phase == "guard":
            nested_submission()

    def checkpoint(event: AgentEvent) -> None:
        if callback_phase == "checkpoint":
            nested_submission()
        accepted.append(event)

    runtime = AgentRuntime(
        2,
        pre_admission_guard=guard,
        admission_checkpoint=checkpoint,
        allow_volatile=True,
    )
    runtime.start()
    try:
        first = runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "outer"
        ).result(timeout=_TIMEOUT)
        second = runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "next"
        ).result(timeout=_TIMEOUT)
        assert runtime.status is AgentRuntimeStatus.ACCEPTING
    finally:
        runtime.shutdown()

    assert not nested_handler_calls
    assert len(blocked) == 1
    assert blocked[0].processing_sequence is None
    assert [event.event_id for event in accepted] == [first.event.event_id, second.event.event_id]
    assert guard_calls == accepted
    assert [first.event.processing_sequence, second.event.processing_sequence] == [1, 2]
    assert first.event.requested_at < blocked[0].requested_at < second.event.requested_at


class _SequencedClock:
    """Return a deterministic, globally ordered timestamp per runtime submit."""

    samples: list[datetime] = []
    values: tuple[datetime, ...] | None = None
    _lock = Lock()

    @classmethod
    def reset(cls, values: tuple[datetime, ...] | None = None) -> None:
        with cls._lock:
            cls.samples = []
            cls.values = values

    @classmethod
    def now(cls, tz: object = None) -> datetime:
        assert tz is UTC
        with cls._lock:
            sample_index = len(cls.samples)
            value = (
                _FIRST_REQUEST_AT + timedelta(seconds=sample_index)
                if cls.values is None
                else cls.values[sample_index]
            )
            cls.samples.append(value)
            return value


class _SubmissionBoundary:
    """Pause producer A before it acquires the runtime's Condition lock."""

    def __init__(self, condition: Condition) -> None:
        self._condition = condition
        self.a_at_queue_boundary = Event()
        self.release_a = Event()
        self.b_admitted = Event()
        self._a_paused = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._condition, name)

    def __enter__(self) -> object:
        if current_thread().name == "producer-A" and not self._a_paused:
            self._a_paused = True
            self.a_at_queue_boundary.set()
            if not self.release_a.wait(timeout=_TIMEOUT):
                raise TimeoutError("producer A was not released at the queue boundary")
        return self._condition.__enter__()

    def __exit__(self, *args: object) -> object:
        return self._condition.__exit__(*args)


class _EventCapture:
    def __init__(self) -> None:
        self._lock = Lock()
        self.events: dict[str, dict[str, AgentEvent]] = {
            phase: {}
            for phase in ("admitted", "started", "handler", "committed", "outcome")
        }
        self._labels_by_id: dict[str, str] = {}
        self.admission_order: list[str] = []

    def accepted(self, event: AgentEvent) -> None:
        label = current_thread().name.removeprefix("producer-")
        with self._lock:
            self._labels_by_id[event.event_id] = label
            self.events["admitted"][label] = event
            self.admission_order.append(label)

    def record(self, phase: str, event: AgentEvent, label: str | None = None) -> None:
        with self._lock:
            resolved_label = label or self._labels_by_id[event.event_id]
            self.events[phase][resolved_label] = event

    def started(self, event: AgentEvent) -> None:
        self.record("started", event)

    def committed(self, event: AgentEvent) -> None:
        self.record("committed", event)


def _install_clock(
    monkeypatch: pytest.MonkeyPatch,
    values: tuple[datetime, ...] | None = None,
) -> None:
    _SequencedClock.reset(values)
    monkeypatch.setattr(agent_runtime_module, "datetime", _SequencedClock)


def _make_runtime(capture: _EventCapture, initial_sequence: int) -> AgentRuntime:
    return AgentRuntime(
        2,
        initial_sequence=initial_sequence,
        admission_checkpoint=capture.accepted,
        started_checkpoint=capture.started,
        internal_commit_checkpoint=capture.committed,
        allow_volatile=True,
    )


def _submit_forced_reverse_order(
    runtime: AgentRuntime,
    handlers: dict[str, Callable[[], object]],
) -> tuple[
    dict[str, Future[AgentEventOutcome[object]]],
    _SubmissionBoundary,
]:
    boundary = _SubmissionBoundary(runtime._condition)
    runtime._condition = boundary  # type: ignore[assignment]
    runtime.start()

    futures: dict[str, Future[AgentEventOutcome[object]]] = {}
    errors: list[BaseException] = []
    result_lock = Lock()

    def submit(label: str) -> None:
        try:
            future = runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                handlers[label],
            )
            with result_lock:
                futures[label] = future
            if label == "B":
                boundary.b_admitted.set()
        except BaseException as error:
            with result_lock:
                errors.append(error)

    producer_a = Thread(target=submit, args=("A",), name="producer-A")
    producer_b = Thread(target=submit, args=("B",), name="producer-B")
    started: list[Thread] = []
    try:
        producer_a.start()
        started.append(producer_a)
        assert boundary.a_at_queue_boundary.wait(timeout=_TIMEOUT)
        producer_b.start()
        started.append(producer_b)
        assert boundary.b_admitted.wait(timeout=_TIMEOUT)
    finally:
        boundary.release_a.set()
        for producer in started:
            producer.join(timeout=_TIMEOUT)

    assert all(not producer.is_alive() for producer in started)
    assert not errors
    assert set(futures) == {"A", "B"}
    return futures, boundary


def _event_identity(event: AgentEvent) -> tuple[object, ...]:
    return (event.event_id, event.event_type, event.source, event.requested_at)


def _collect_outcomes(
    futures: dict[str, Future[AgentEventOutcome[object]]], capture: _EventCapture
) -> dict[str, AgentEventOutcome[object]]:
    outcomes: dict[str, AgentEventOutcome[object]] = {}
    for label in ("B", "A"):
        outcome = futures[label].result(timeout=_TIMEOUT)
        outcomes[label] = outcome
        capture.record("outcome", outcome.event, label)
    return outcomes


def _assert_lifecycle_identity(
    capture: _EventCapture,
    outcomes: dict[str, AgentEventOutcome[object]],
    initial_sequence: int,
) -> None:
    for sequence, label in enumerate(("B", "A"), start=initial_sequence + 1):
        admitted = capture.events["admitted"][label]
        outcome_event = outcomes[label].event
        assert admitted.processing_sequence is None
        assert _event_identity(admitted) == _event_identity(outcome_event)
        assert outcome_event.processing_sequence == sequence
        for phase in ("started", "handler", "committed", "outcome"):
            assert capture.events[phase][label] is outcome_event


def test_runtime_fifo_admission_has_nondecreasing_request_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    capture = _EventCapture()
    initial_sequence = 40
    runtime = _make_runtime(capture, initial_sequence)
    executed: list[str] = []

    def handler(label: str) -> Callable[[], str]:
        def run() -> str:
            event = runtime.current_event()
            assert event is not None
            capture.record("handler", event, label)
            executed.append(label)
            return label

        return run

    boundary: _SubmissionBoundary | None = None
    try:
        futures, boundary = _submit_forced_reverse_order(
            runtime,
            {"A": handler("A"), "B": handler("B")},
        )
        outcomes = _collect_outcomes(futures, capture)
    finally:
        if boundary is not None:
            boundary.release_a.set()
        runtime.shutdown()

    assert capture.admission_order == ["B", "A"]
    assert executed == ["B", "A"]
    assert [outcomes[label].value for label in ("B", "A")] == [
        "B",
        "A",
    ]
    accepted_times = tuple(
        capture.events["admitted"][label].requested_at for label in ("B", "A")
    )
    assert accepted_times == tuple(sorted(accepted_times))
    assert accepted_times == (
        _FIRST_REQUEST_AT,
        _FIRST_REQUEST_AT + timedelta(seconds=1),
    )
    assert accepted_times == tuple(_SequencedClock.samples)
    _assert_lifecycle_identity(capture, outcomes, initial_sequence)


def _goal_proposal_for_event(event: AgentEvent) -> GoalRecord:
    assert event.processing_sequence is not None
    target = R13Reference(R13ReferenceKind.VALUE, "value:ordered-time")
    origin_refs = (
        R13Reference(R13ReferenceKind.USER_REQUEST, "request:ordered-time"),
    )
    evidence_refs = (
        R13Reference(R13ReferenceKind.EVENT, "event:ordered-time-proof"),
    )
    description = "preserve the ordered runtime timestamp"
    proposal_digest = goal_proposal_digest(
        target,
        description,
        origin_refs=origin_refs,
        evidence_refs=evidence_refs,
    )
    goal_id = goal_id_for_target(target)
    genesis = GoalRevisionRecord(
        goal_id=goal_id,
        revision=0,
        operation=GoalRevisionOperation.CREATE,
        reason=GoalRevisionReason.CREATION,
        created_at=event.requested_at,
        previous_lifecycle_state=None,
        proposal_digest=proposal_digest,
        event_id=event.event_id,
        event_sequence=event.processing_sequence,
        evidence_refs=tuple(sorted(reference.reference for reference in evidence_refs)),
    )
    return GoalRecord(
        goal_id=goal_id,
        target=target,
        description=description,
        origin_refs=origin_refs,
        evidence_refs=evidence_refs,
        revision_history=(genesis,),
    )


def _goal_event(event: AgentEvent, proposal: GoalRecord) -> GoalMutationEvidence:
    assert event.processing_sequence is not None
    return GoalMutationEvidence(
        event_id=event.event_id,
        event_sequence=event.processing_sequence,
        recorded_at=event.requested_at,
        evidence_refs=proposal.evidence_refs,
    )


def test_real_chat_attention_belief_and_goal_keep_runtime_event_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    settings = _settings(tmp_path)
    provider = _RecordingProvider()
    memory = DualMemorySystem(settings)
    monkeypatch.setattr(memory, "retrieve_context", lambda _query: MemoryContext())
    loop = SuzkaMainLoop(settings, provider, memory)
    capture = _EventCapture()
    initial_sequence = 80
    runtime = _make_runtime(capture, initial_sequence)
    loop.bind_runtime(runtime)
    executed: list[str] = []
    belief_proposal: dict[str, BeliefRecord] = {}
    goal_proposal: dict[str, GoalRecord] = {}
    attention_events: list[tuple[AgentEvent, AttentionEvent]] = []

    original_refresh = AttentionSystem.refresh

    def refresh_spy(
        owner: AttentionSystem,
        projections: tuple[object, ...],
        event: AttentionEvent,
        *,
        global_emotion: object = None,
    ) -> AttentionRefreshResult:
        active_event = runtime.current_event()
        assert active_event is not None
        attention_events.append((active_event, event))
        return original_refresh(
            owner,
            projections,  # type: ignore[arg-type]
            event,
            global_emotion=global_emotion,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(AttentionSystem, "refresh", refresh_spy)

    def handler(label: str) -> Callable[[], object]:
        def run() -> object:
            event = runtime.current_event()
            assert event is not None
            assert event.processing_sequence is not None
            capture.record("handler", event, label)
            executed.append(label)
            result = loop.chat(f"ordered timestamp chat {label}")

            if label == "B":
                belief_proposal["record"] = loop._belief_system.create_proposal(
                    BeliefProposition("ordered runtime event belief"),
                    evidence=(
                        BeliefEvidence(
                            "claim:ordered-runtime-event",
                            BeliefEvidenceType.EXTERNAL_CLAIM,
                        ),
                    ),
                )
                proposal = _goal_proposal_for_event(event)
                goal_proposal["record"] = proposal
                loop._goal_system.ingest_proposal(
                    proposal,
                    _goal_event(event, proposal),
                )
            else:
                proposal = belief_proposal["record"]
                admission = BeliefSubjectAdmission(
                    proposal.proposition.proposition_digest,
                    tuple(
                        sorted(
                            item.evidence_ref for item in proposal.evidence
                        )
                    ),
                    event.event_id,
                    event.processing_sequence,
                    AdmissionReason.SUBJECT_ENDORSEMENT,
                )
                loop._belief_system.adopt(proposal.belief_id, admission)

                goal = goal_proposal["record"]
                goal_admission = GoalSubjectAdmission(
                    goal_id=goal.goal_id,
                    proposal_digest=goal.proposal_digest,
                    evidence_refs=tuple(
                        sorted(reference.reference for reference in goal.evidence_refs)
                    ),
                    event_id=event.event_id,
                    event_sequence=event.processing_sequence,
                    reason=GoalAdmissionReason.SUBJECT_ENDORSEMENT,
                )
                loop._goal_system.adopt(
                    goal.goal_id,
                    goal_admission,
                    _goal_event(event, goal),
                )
            return result

        return run

    boundary: _SubmissionBoundary | None = None
    try:
        futures, boundary = _submit_forced_reverse_order(
            runtime,
            {"A": handler("A"), "B": handler("B")},
        )
        outcomes = _collect_outcomes(futures, capture)
    finally:
        if boundary is not None:
            boundary.release_a.set()
        runtime.shutdown()

    assert capture.admission_order == ["B", "A"]
    assert executed == ["B", "A"]
    _assert_lifecycle_identity(capture, outcomes, initial_sequence)
    assert len(provider.prompts) == 2
    assert "ordered timestamp chat B" in provider.prompts[0]
    assert "ordered timestamp chat A" in provider.prompts[1]

    assert [active.event_id for active, _attention in attention_events] == [
        outcomes[label].event.event_id
        for label in ("B", "A")
    ]
    for label, (active, attention) in zip(
        ("B", "A"), attention_events, strict=True
    ):
        runtime_event = outcomes[label].event
        assert active is capture.events["handler"][label]
        assert attention.event_id == runtime_event.event_id
        assert attention.event_sequence == runtime_event.processing_sequence
        assert attention.occurred_at == runtime_event.requested_at

    admitted_times = tuple(
        capture.events["admitted"][label].requested_at for label in ("B", "A")
    )
    assert admitted_times == tuple(sorted(admitted_times))
    assert admitted_times == tuple(_SequencedClock.samples)

    belief = loop._belief_system.records[0]
    proposal_event = outcomes["B"].event
    adoption_event = outcomes["A"].event
    assert belief.revision_history[0].event_id == proposal_event.event_id
    assert belief.revision_history[0].event_sequence == proposal_event.processing_sequence
    assert belief.revision_history[0].created_at == proposal_event.requested_at
    assert belief.subject_admission is not None
    assert belief.subject_admission.event_id == adoption_event.event_id
    assert belief.subject_admission.event_sequence == adoption_event.processing_sequence
    assert belief.revision_history[-1].created_at == adoption_event.requested_at

    goal = loop._goal_system.get(goal_proposal["record"].goal_id)
    assert goal is not None
    assert goal.revision_history[0].event_id == proposal_event.event_id
    assert goal.revision_history[0].event_sequence == proposal_event.processing_sequence
    assert goal.revision_history[0].created_at == proposal_event.requested_at
    assert goal.revision_history[-1].event_id == adoption_event.event_id
    assert goal.revision_history[-1].event_sequence == adoption_event.processing_sequence
    assert goal.revision_history[-1].created_at == adoption_event.requested_at
    goal_receipts = loop._goal_system.snapshot().event_receipts
    assert [receipt.operation for receipt in goal_receipts] == [
        GoalSystemEventOperation.INGEST_PROPOSAL,
        GoalSystemEventOperation.ADOPT,
    ]
    assert [receipt.recorded_at for receipt in goal_receipts] == [
        proposal_event.requested_at,
        adoption_event.requested_at,
    ]
    assert [
        (receipt.event_id, receipt.event_sequence)
        for receipt in goal_receipts
    ] == [
        (proposal_event.event_id, proposal_event.processing_sequence),
        (adoption_event.event_id, adoption_event.processing_sequence),
    ]


def test_backward_clock_refusal_preserves_admission_boundary_and_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t1 = _FIRST_REQUEST_AT
    t2 = t1 + timedelta(seconds=1)
    t3 = t1 + timedelta(seconds=2)
    _install_clock(monkeypatch, (t2, t1, t2, t3))
    guard_events: list[AgentEvent] = []
    admitted_events: list[AgentEvent] = []
    handled_events: list[AgentEvent] = []

    def guard(event: AgentEvent) -> bool:
        guard_events.append(event)
        return True

    runtime = AgentRuntime(
        4,
        pre_admission_guard=guard,
        admission_checkpoint=admitted_events.append,
        allow_volatile=True,
    )

    def handler(label: str) -> Callable[[], str]:
        def run() -> str:
            event = runtime.current_event()
            assert event is not None
            handled_events.append(event)
            return label

        return run

    runtime.start()
    try:
        first = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            handler("first"),
        )
        with pytest.raises(AgentRuntimeAdmissionBlocked) as blocked:
            runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                handler("backward"),
            )

        rejected_event = blocked.value.event
        assert rejected_event.requested_at == t1
        assert rejected_event.processing_sequence is None
        assert rejected_event.event_type is AgentEventType.CHAT
        assert rejected_event.source is AgentEventSource.API_CHAT
        assert runtime.status is AgentRuntimeStatus.ACCEPTING
        assert len(guard_events) == 1
        assert admitted_events == [guard_events[0]]

        equal_time = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            handler("equal-time"),
        )
        later_time = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            handler("later-time"),
        )
        outcomes = [
            first.result(timeout=_TIMEOUT),
            equal_time.result(timeout=_TIMEOUT),
            later_time.result(timeout=_TIMEOUT),
        ]
        assert runtime.status is AgentRuntimeStatus.ACCEPTING
    finally:
        runtime.shutdown()

    assert tuple(_SequencedClock.samples) == (t2, t1, t2, t3)
    assert [event.requested_at for event in admitted_events] == [t2, t2, t3]
    assert [event.requested_at for event in guard_events] == [t2, t2, t3]
    assert [event.processing_sequence for event in admitted_events] == [None, None, None]
    assert [outcome.event.processing_sequence for outcome in outcomes] == [1, 2, 3]
    assert [outcome.event.requested_at for outcome in outcomes] == [t2, t2, t3]
    assert [outcome.value for outcome in outcomes] == [
        "first",
        "equal-time",
        "later-time",
    ]
    assert [event.event_id for event in handled_events] == [
        outcome.event.event_id for outcome in outcomes
    ]
    assert rejected_event.event_id not in {
        event.event_id for event in guard_events + admitted_events + handled_events
    }


def test_false_pre_admission_guard_does_not_advance_time_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t1 = _FIRST_REQUEST_AT
    t2 = t1 + timedelta(seconds=1)
    t3 = t1 + timedelta(seconds=2)
    _install_clock(monkeypatch, (t1, t3, t2))
    guard_events: list[AgentEvent] = []
    admitted_events: list[AgentEvent] = []
    handled_events: list[AgentEvent] = []

    def guard(event: AgentEvent) -> bool:
        guard_events.append(event)
        return event.requested_at != t3

    runtime = AgentRuntime(
        3,
        pre_admission_guard=guard,
        admission_checkpoint=admitted_events.append,
        allow_volatile=True,
    )

    def handler(label: str) -> Callable[[], str]:
        def run() -> str:
            event = runtime.current_event()
            assert event is not None
            handled_events.append(event)
            return label

        return run

    runtime.start()
    try:
        first = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            handler("first"),
        )
        with pytest.raises(AgentRuntimeAdmissionBlocked) as blocked:
            runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                handler("guard-refused"),
            )
        rejected_event = blocked.value.event
        assert rejected_event.requested_at == t3
        assert rejected_event.processing_sequence is None
        assert runtime.status is AgentRuntimeStatus.ACCEPTING

        later = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            handler("after-refusal"),
        )
        outcomes = [
            first.result(timeout=_TIMEOUT),
            later.result(timeout=_TIMEOUT),
        ]
    finally:
        runtime.shutdown()

    assert tuple(_SequencedClock.samples) == (t1, t3, t2)
    assert [event.requested_at for event in guard_events] == [t1, t3, t2]
    assert [event.requested_at for event in admitted_events] == [t1, t2]
    assert [event.processing_sequence for event in admitted_events] == [None, None]
    assert [outcome.event.processing_sequence for outcome in outcomes] == [1, 2]
    assert [outcome.event.requested_at for outcome in outcomes] == [t1, t2]
    assert [outcome.value for outcome in outcomes] == ["first", "after-refusal"]
    assert [event.event_id for event in handled_events] == [
        outcome.event.event_id for outcome in outcomes
    ]
    assert rejected_event.event_id not in {
        event.event_id for event in admitted_events + handled_events
    }
