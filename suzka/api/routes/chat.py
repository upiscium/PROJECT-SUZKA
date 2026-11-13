"""Chat routes."""

from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request

from suzka.api.dependencies import get_agent_runtime, get_main_loop
from suzka.api.runtime_execution import execute_with_event
from suzka.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CommandTicketSchema,
    CommandTicketStatusResponse,
    EmotionSchema,
    ModelSchema,
)
from suzka.r16.driver import R16Driver
from suzka.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    ChatResult,
    ChatContextSelectors,
    SuzkaMainLoop,
)


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    http: Request,
    main_loop: SuzkaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatResponse:
    reject_unsupported_attachments(request)
    selectors = ChatContextSelectors(
        context_id=request.context_id,
        client_session_id=request.client_session_id,
        interlocutor_key=request.interlocutor_key,
    )
    outcome = execute_with_event(
        runtime,
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: main_loop.chat(request.message, selectors=selectors),
    )
    result = cast(ChatResult, outcome.value)
    response = chat_response_from_result(result)
    driver = _driver(http)
    if driver is None:
        return response
    # The one CHAT hook (#370 D0 §1): enqueue only, right before responding.
    ticket = driver.notify_chat_terminal(
        outcome,  # type: ignore[arg-type]
        request.message,
    )
    return response.model_copy(
        update={
            "incarnation": driver.incarnation if driver.active else None,
            "command_ticket": None
            if ticket is None
            else CommandTicketSchema(
                ticket_id=ticket.ticket_id,
                kind=ticket.kind.value,
                status=ticket.status.value,
            ),
        }
    )


@router.get(
    "/chat/commands/{ticket_id}", response_model=CommandTicketStatusResponse
)
def command_status(ticket_id: str, http: Request) -> CommandTicketStatusResponse:
    """A command ticket's settled status (#370 D1 2-2); 404 when unknown.

    Same authentication as chat. It reads only the driver's in-memory
    results, never live owner state (D2 P1-4).
    """

    driver = _driver(http)
    view = None if driver is None else driver.ticket(ticket_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Command ticket not found")
    return CommandTicketStatusResponse(
        status=view.status.value,
        reason=None if view.reason is None else view.reason.value,
    )


def _driver(http: Request) -> R16Driver | None:
    driver = getattr(http.app.state, "r16_driver", None)
    return driver if isinstance(driver, R16Driver) else None


def reject_unsupported_attachments(request: ChatRequest) -> None:
    if request.attachments:
        raise HTTPException(
            status_code=422,
            detail="Attachments are schema-only in v1.0; runtime execution is text-only.",
        )


def chat_response_from_result(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        episode_id=result.episode_id,
        response=result.response,
        emotion=EmotionSchema(
            valence=result.valence,
            arousal=result.arousal,
            optimal_loss=result.optimal_loss,
        ),
        model=ModelSchema(model_id=result.model_id, adapter_id=result.adapter_id),
        experience_id=result.experience_id,
    )
