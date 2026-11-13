"""R16 operator routes (#370 D0 §6, IP-6)."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from suzka.api.dependencies import get_agent_runtime, require_admin
from suzka.api.runtime_execution import execute
from suzka.r16.activation import R16DriverAlreadyActive, activation_handler
from suzka.runtime import AgentEventSource, AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/r16", tags=["r16"], dependencies=[Depends(require_admin)]
)


class R16DriverActivationResponse(BaseModel):
    active: bool
    activation_sequence: int


@router.post("/driver/activate", response_model=R16DriverActivationResponse)
def activate_driver(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> R16DriverActivationResponse:
    """Record the unique R16 driver activation; a second call is 409."""

    try:
        sequence = execute(
            runtime,
            AgentEventType.R16_DRIVER_ACTIVATION,
            AgentEventSource.API_R16_DRIVER_ACTIVATE,
            activation_handler(request.app.state.event_journal, runtime),
        )
    except R16DriverAlreadyActive as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="R16 driver is already activated",
        ) from exc
    driver = getattr(request.app.state, "r16_driver", None)
    if driver is not None:
        driver.activate()
    return R16DriverActivationResponse(active=True, activation_sequence=sequence)
