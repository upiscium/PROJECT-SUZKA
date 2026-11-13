"""Chat API schemas."""

from pydantic import BaseModel, Field, StrictStr, field_validator

from suzka.identifiers import validate_identifier


class AttachmentSchema(BaseModel):
    type: str = Field(min_length=1)
    url: str | None = None
    name: str | None = None
    content_type: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    attachments: list[AttachmentSchema] = Field(default_factory=list)
    debug: bool = False
    context_id: StrictStr | None = None
    client_session_id: StrictStr | None = None
    interlocutor_key: StrictStr | None = None

    @field_validator("context_id", "client_session_id", "interlocutor_key")
    @classmethod
    def validate_selector(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_identifier(value)
        except Exception:
            raise ValueError("selector must be a valid opaque identifier") from None


class EmotionSchema(BaseModel):
    valence: float
    arousal: float
    optimal_loss: float


class ModelSchema(BaseModel):
    model_id: str
    adapter_id: str | None = None


class CommandTicketSchema(BaseModel):
    ticket_id: str
    kind: str
    status: str


class ChatResponse(BaseModel):
    episode_id: str
    response: str
    emotion: EmotionSchema
    model: ModelSchema
    # DRV-3 (#370): the turn's Experience, so the Chat screen can offer
    # "review this" / "commit to review this"; None when none was created.
    experience_id: str | None = None
    # IP-6b (#370 D1 2-1): a command turn's ticket, and the server
    # incarnation, so the screen can tell that earlier turns predate a
    # restart (D2 addendum P2-3). Both None while the driver is inactive.
    command_ticket: CommandTicketSchema | None = None
    incarnation: str | None = None


class CommandTicketStatusResponse(BaseModel):
    status: str
    reason: str | None = None
