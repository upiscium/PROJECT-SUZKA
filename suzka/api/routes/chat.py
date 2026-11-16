"""Chat routes."""

import logging
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request

from suzka.api.dependencies import (
    get_agent_runtime,
    get_chat_log,
    get_main_loop,
    open_chat_log,
)
from suzka.api.runtime_execution import execute_with_event
from suzka.api.schemas.chat import (
    ChatLogTurnSchema,
    ChatRequest,
    ChatResponse,
    ChatSessionCreateRequest,
    ChatSessionDetailResponse,
    ChatSessionListResponse,
    ChatSessionRenameRequest,
    ChatSessionSchema,
    CommandTicketSchema,
    CommandTicketStatusResponse,
    EmotionSchema,
    ModelSchema,
)
from suzka.chat_log import ChatLogStore, ChatLogUnavailable, ChatSession, ChatSessionNotFound
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
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    http: Request,
    main_loop: SuzkaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatResponse:
    reject_unsupported_attachments(request)
    # The transcript is opened only for a request that names a session, and
    # an unusable store never blocks the turn (it is display-only).
    chat_log: ChatLogStore | None = None
    if request.chat_session_id is not None:
        try:
            chat_log = open_chat_log(http)
            # An unknown session fails before the turn is admitted.
            _require_session(chat_log, request.chat_session_id)
        except ChatLogUnavailable:
            logger.warning("chat transcript unavailable; the turn runs unsaved", exc_info=True)
            chat_log = None
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
    if request.chat_session_id is not None:
        response = response.model_copy(
            update={
                "transcript_saved": chat_log is not None
                and _save_exchange(chat_log, request.chat_session_id, request.message, result)
            }
        )
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


@router.get("/chat/sessions", response_model=ChatSessionListResponse)
def list_chat_sessions(
    chat_log: ChatLogStore = Depends(get_chat_log),
) -> ChatSessionListResponse:
    return ChatSessionListResponse(
        sessions=[_session_schema(session) for session in chat_log.list_sessions()]
    )


@router.post("/chat/sessions", response_model=ChatSessionSchema, status_code=201)
def create_chat_session(
    request: ChatSessionCreateRequest,
    chat_log: ChatLogStore = Depends(get_chat_log),
) -> ChatSessionSchema:
    return _session_schema(chat_log.create_session(request.title))


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionDetailResponse)
def get_chat_session(
    session_id: str, chat_log: ChatLogStore = Depends(get_chat_log)
) -> ChatSessionDetailResponse:
    session = _require_session(chat_log, session_id)
    return ChatSessionDetailResponse(
        session=_session_schema(session),
        turns=[
            ChatLogTurnSchema(
                role=turn.role,
                content=turn.content,
                episode_id=turn.episode_id,
                created_at=turn.created_at,
            )
            for turn in chat_log.turns(session_id)
        ],
    )


@router.patch("/chat/sessions/{session_id}", response_model=ChatSessionSchema)
def rename_chat_session(
    session_id: str,
    request: ChatSessionRenameRequest,
    chat_log: ChatLogStore = Depends(get_chat_log),
) -> ChatSessionSchema:
    try:
        return _session_schema(chat_log.rename_session(session_id, request.title))
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Chat session not found") from None


@router.delete("/chat/sessions/{session_id}", status_code=204)
def delete_chat_session(
    session_id: str, chat_log: ChatLogStore = Depends(get_chat_log)
) -> None:
    """Delete a session's transcript only; what the subject remembers is untouched."""

    try:
        chat_log.delete_session(session_id)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Chat session not found") from None


def _require_session(chat_log: ChatLogStore, session_id: str) -> ChatSession:
    try:
        return chat_log.get_session(session_id)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Chat session not found") from None


def _save_exchange(
    chat_log: ChatLogStore, session_id: str, message: str, result: ChatResult
) -> bool:
    """Record a committed turn; a failure here never fails the turn itself."""

    try:
        chat_log.append_exchange(session_id, message, result.response, result.episode_id)
    except Exception:
        logger.warning("chat transcript write failed", exc_info=True)
        return False
    return True


def _session_schema(session: ChatSession) -> ChatSessionSchema:
    return ChatSessionSchema(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
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
