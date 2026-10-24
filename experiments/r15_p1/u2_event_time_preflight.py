"""TEST ONLY: event-time R09 participant snapshot feasibility for R15 U2.

A captured ContextFrame is a volatile experimental observation, NOT a
persisted owner witness, identity binding, relationship admission, or
authoritative proof of who actually spoke. U2 production remains UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from suzka.memory import DualMemorySystem, ExperienceStore
from suzka.memory.experience_participant import experience_id_for_event
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.chat_context import ChatContextSelectors
from suzka.runtime.context import (
    ContextFrame,
    ContextRegistry,
    ContextStatus,
)
from suzka.runtime.event_journal import (
    EventJournal,
    EventLifecycle,
)


class EventTimeStatus(str, Enum):
    OPAQUE_OBSERVED = "test_only_one_opaque_ref_observed_in_chat_event"
    HOLD = "hold_event_time_or_source_unavailable"


@dataclass(frozen=True, slots=True)
class EventTimeObservation:
    """Volatile immutable R09 frame actually selected during an R07 Chat."""

    event_id: str
    event_sequence: int
    frame: ContextFrame
    registry_revision: int


@dataclass(frozen=True, slots=True)
class EventTimeParticipantView:
    """No Person verification, trust, narrative, confidence or acceptance."""

    status: EventTimeStatus
    participant_ref: str | None
    event_context_revision: int | None
    person_verified: bool = False
    relationship_admitted: bool = False


_HOLD = EventTimeParticipantView(EventTimeStatus.HOLD, None, None)


class EventTimeCapture:
    """A test-only wrapper for real MainLoop.resolve_chat_context.

    Install ONLY through a pytest monkeypatch around an isolated TestClient;
    it records the immutable ContextFrame during the actual R07 worker event.
    The captured frame is not durable in ExperienceRecord or StateWAL.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        resolver: Callable[[ContextRegistry, ChatContextSelectors], ContextFrame],
    ) -> None:
        if type(runtime) is not AgentRuntime:
            raise TypeError("test capture requires the actual isolated runtime")
        self._runtime = runtime
        self._resolver = resolver
        self.observations: list[EventTimeObservation] = []

    def __call__(
        self,
        registry: ContextRegistry,
        selectors: ChatContextSelectors,
    ) -> ContextFrame:
        event = self._runtime.current_event()
        if (
            event is None
            or event.event_type is not AgentEventType.CHAT
            or event.source is not AgentEventSource.API_CHAT
            or type(event.processing_sequence) is not int
        ):
            raise RuntimeError("no ordinary serialized Chat at event-time Context read")
        frame = self._resolver(registry, selectors)
        self.observations.append(
            EventTimeObservation(
                event.event_id,
                event.processing_sequence,
                frame,
                registry.state.revision,
            )
        )
        return frame


def assess_single_opaque_participant(
    observation: EventTimeObservation,
    episode_id: str,
    journal: EventJournal,
    experience: ExperienceStore,
    memory: DualMemorySystem,
    contexts: ContextRegistry,
) -> EventTimeParticipantView:
    """Cooperative test read of actual terminal records, never U2 authority.

    A changed R09 frame, missing R12/DB1, ambiguous participant or forged
    event tuple returns HOLD; never infer retroactive or verified person.
    """
    if (
        type(observation) is not EventTimeObservation
        or type(journal) is not EventJournal
        or type(experience) is not ExperienceStore
        or type(memory) is not DualMemorySystem
        or type(contexts) is not ContextRegistry
        or type(episode_id) is not str
    ):
        return _HOLD
    frame = observation.frame
    if (
        frame.status is not ContextStatus.ACTIVE
        or len(frame.participant_refs) != 1
        or type(observation.event_sequence) is not int
        or observation.event_sequence < 1
        or frame.last_modified_revision > observation.registry_revision
    ):
        return _HOLD

    try:
        if contexts.get(frame.context_id) != frame:
            # Includes all late participant additions, closure, revisions.
            return _HOLD
        relevant = [
            row
            for row in journal.inspect().records
            if row.event_id == observation.event_id
        ]
        if any(
            row.event_type is not AgentEventType.CHAT
            or row.source is not AgentEventSource.API_CHAT
            or row.processing_sequence not in (None, observation.event_sequence)
            for row in relevant
        ) or any(row.lifecycle is EventLifecycle.FAILED for row in relevant):
            return _HOLD
        # ACCEPTED has no allocated processing sequence yet; the later
        # TRANSACTION_COMPLETED/COMPLETED records MUST have the exact one.
        if (
            sum(
                row.lifecycle is EventLifecycle.TRANSACTION_COMPLETED
                and row.processing_sequence == observation.event_sequence
                for row in relevant
            )
            != 1
            or sum(
                row.lifecycle is EventLifecycle.COMPLETED
                and row.processing_sequence == observation.event_sequence
                for row in relevant
            )
            != 1
        ):
            return _HOLD
        for status in (
            EventLifecycle.ACCEPTED,
            EventLifecycle.TRANSACTION_COMPLETED,
            EventLifecycle.COMPLETED,
        ):
            if sum(row.lifecycle is status for row in relevant) != 1:
                return _HOLD
        entry = experience.load_committed_current(
            experience_id_for_event(
                observation.event_id,
                observation.event_sequence,
            )
        )
        owned = entry.record
        if (
            owned.source_event_id != observation.event_id
            or owned.source_event_sequence != observation.event_sequence
            or owned.source_episode_id != episode_id
            or owned.context_id != frame.context_id
        ):
            return _HOLD
        episodic = memory.get_committed_episodic(episode_id)
        if episodic is None:
            return _HOLD
        record = episodic.record
        if (
            record.context_id != frame.context_id
            or record.source_channel != frame.source_channel
            or record.source_session_id != frame.source_session_id
        ):
            return _HOLD
    except (RuntimeError, ValueError, TypeError, OSError, AttributeError):
        return _HOLD

    return EventTimeParticipantView(
        EventTimeStatus.OPAQUE_OBSERVED,
        frame.participant_refs[0],
        frame.last_modified_revision,
    )
