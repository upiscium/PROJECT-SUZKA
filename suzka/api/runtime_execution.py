"""Small boundary between HTTP handlers and the serialized agent runtime."""

from collections.abc import Callable
from typing import TypeVar, overload

from fastapi import HTTPException, status

from suzka.identity import ValueDomainError, ValueNotFound
from suzka.runtime import (
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeDurabilityError,
    AgentRuntimeExecutionError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    CoordinatedResult,
    ContextCapacityExceeded,
    ContextConflict,
    ContextError,
    ContextNotFound,
    ContextStateInvalid,
)


T = TypeVar("T")


@overload
def execute(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], CoordinatedResult[T]],
) -> T: ...


@overload
def execute(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], T],
) -> T: ...


def execute(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], object],
) -> object:
    """Submit one request handler, translating only runtime-boundary failures."""

    return execute_with_event(runtime, event_type, source, handler).value


def execute_with_event(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], object],
) -> AgentEventOutcome[object]:
    """Like ``execute``, but also return the admitted event (#370 D0 §1).

    The R16 driver needs the CHAT event's id and processing sequence to
    name the turn as a source; the HTTP error mapping is identical.
    """

    try:
        return runtime.submit(event_type, source, handler).result()
    except (
        AgentRuntimeAdmissionBlocked,
        AgentRuntimeQueueFull,
        AgentRuntimeStopped,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runtime is temporarily unavailable",
        ) from exc
    except AgentRuntimeDurabilityError as exc:
        if exc.outcome_indeterminate:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent mutation durability is indeterminate",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runtime durability is temporarily unavailable",
        ) from exc
    except AgentRuntimeExecutionError as exc:
        cause = exc.__cause__
        if isinstance(cause, ValueNotFound):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Value not found",
            ) from exc
        if isinstance(cause, ValueDomainError):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Value operation is invalid",
            ) from exc
        if isinstance(cause, ContextNotFound):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Context not found",
            ) from exc
        if isinstance(
            cause, (ContextConflict, ContextStateInvalid, ContextCapacityExceeded)
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Context selection is invalid",
            ) from exc
        if isinstance(cause, ContextError):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Context operation is invalid",
            ) from exc
        if exc.__cause__ is not None:
            raise exc.__cause__
        raise
