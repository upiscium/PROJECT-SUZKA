"""R16 driver activation marker (#370 D0 §6, IP-6).

The driver runs only after an operator has recorded the unique
``R16_DRIVER_ACTIVATION`` event. Its completed record is the marker. It is
looked up in the retained Journal and, once archived, in the records the
compaction in effect carries (C9, Human decision 2026-11-13), so activation
survives the retention window. This module grants no authority beyond
"the driver may select and submit"; owners still decide.
"""

from __future__ import annotations

from collections.abc import Callable

from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalInspection,
    EventJournalRecord,
    EventLifecycle,
)


class R16DriverAlreadyActive(ValueError):
    """The unique activation already exists; a second one is refused."""


def driver_activation(inspection: EventJournalInspection) -> EventJournalRecord | None:
    """The completed activation record, retained or carried, if any."""

    for record in (*reversed(inspection.records), *inspection.carried_records):
        if (
            record.lifecycle is EventLifecycle.COMPLETED
            and record.event_type is AgentEventType.R16_DRIVER_ACTIVATION
            and record.source is AgentEventSource.API_R16_DRIVER_ACTIVATE
        ):
            return record
    return None


def activation_handler(
    journal: EventJournal, runtime: AgentRuntime
) -> Callable[[], int]:
    """The serialized handler of one activation event.

    Uniqueness is checked inside the runtime, so two requests cannot both
    pass: the earlier one has completed before the later handler runs.
    Returns the activation event's processing sequence.
    """

    def handle() -> int:
        if driver_activation(journal.inspect()) is not None:
            raise R16DriverAlreadyActive("R16 driver is already activated")
        event = runtime.current_event()
        if (
            event is None
            or event.event_type is not AgentEventType.R16_DRIVER_ACTIVATION
            or event.source is not AgentEventSource.API_R16_DRIVER_ACTIVATE
            or event.processing_sequence is None
        ):
            raise RuntimeError("activation runs only as its own event")
        return event.processing_sequence

    return handle


class R16SelfGoalAlreadyActive(ValueError):
    """IP-11: the unique self-Goal activation already exists."""


def self_goal_activation(
    inspection: EventJournalInspection,
) -> EventJournalRecord | None:
    """IP-11 (#379 D2 P2-2 / D3 P2-3): the completed self-Goal marker.

    Retained, or carried by the compaction in effect (C11).  Only
    Motivations from U2a events after this marker can ground a self Goal.
    """

    for record in (*reversed(inspection.records), *inspection.carried_records):
        if (
            record.lifecycle is EventLifecycle.COMPLETED
            and record.event_type is AgentEventType.R16_SELF_GOAL_ACTIVATION
            and record.source is AgentEventSource.API_R16_SELF_GOAL_ACTIVATE
        ):
            return record
    return None


def self_goal_activation_handler(
    journal: EventJournal, runtime: AgentRuntime
) -> Callable[[], int]:
    """The serialized handler of the unique self-Goal activation event."""

    def handle() -> int:
        if self_goal_activation(journal.inspect()) is not None:
            raise R16SelfGoalAlreadyActive("self-Goal path is already activated")
        event = runtime.current_event()
        if (
            event is None
            or event.event_type is not AgentEventType.R16_SELF_GOAL_ACTIVATION
            or event.source is not AgentEventSource.API_R16_SELF_GOAL_ACTIVATE
            or event.processing_sequence is None
        ):
            raise RuntimeError("activation runs only as its own event")
        return event.processing_sequence

    return handle
