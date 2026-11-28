"""Debug API schemas."""

from typing import Any

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from suzka.api.schemas.chat import ChatResponse, EmotionSchema
from suzka.body import EmotionUpdateReasonCode
from suzka.cognition import AppraisalReasonCode, LossInvalidReason


class RetrievedEpisodeSchema(BaseModel):
    id: str
    user_input: str
    response: str
    record_type: str


class RetrievedSemanticSchema(BaseModel):
    id: str
    text: str
    record_type: str


class RetrievedMemorySchema(BaseModel):
    db1_results: list[RetrievedEpisodeSchema]
    db2_results: list[RetrievedSemanticSchema]


class GenerationParamsSchema(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


class LossMeasurementSchema(BaseModel):
    model_key: str
    raw_loss: float | None
    valid: bool
    invalid_reason: LossInvalidReason | None
    calibrated_novelty: float | None


class AppraisalSchema(BaseModel):
    novelty: float | None
    novelty_valid: bool
    goal_progress: float | None
    threat: float | None
    controllability: float | None
    certainty: float | None
    social_relevance: float | None
    effort_cost: float | None
    reasons: list[AppraisalReasonCode]


class ValenceContributionsSchema(BaseModel):
    goal_progress: float
    threat: float
    effort_cost: float
    controllability: float


class ArousalContributionsSchema(BaseModel):
    novelty: float
    threat: float
    effort_cost: float
    social_relevance: float
    uncertainty: float
    low_controllability: float


class EmotionUpdateSchema(BaseModel):
    state: EmotionSchema
    valence_contributions: ValenceContributionsSchema
    arousal_contributions: ArousalContributionsSchema
    reasons: list[EmotionUpdateReasonCode]


class ConversationTraceItemSchema(BaseModel):
    episode_id: str
    retention: float
    stability_hours: float
    display: str
    double_recall: bool
    read_status: str


class ConversationTraceDiagnosticsSchema(BaseModel):
    """#417 D1 §6-2: how this turn saw the ConversationTrace (admin debug only)."""

    state: str
    items: list[ConversationTraceItemSchema]


class ChatDiagnosticsSchema(BaseModel):
    measurement: LossMeasurementSchema
    appraisal: AppraisalSchema
    temporal_update: EmotionUpdateSchema
    emotion_update: EmotionUpdateSchema


class ComposedMessageSchema(BaseModel):
    role: str
    content: str


class CompositionReportSchema(BaseModel):
    """#500 3b-5: numbers only; present on non-legacy turns."""

    mode: str
    exchanges_used: int
    exchanges_dropped: int
    state_block_bytes: int
    system_bytes: int
    sections_dropped: int


class DebugChatResponse(ChatResponse):
    hidden_thought: str
    loss: float | None
    prompt: str
    retrieved_memory: RetrievedMemorySchema
    generation_params: GenerationParamsSchema
    diagnostics: ChatDiagnosticsSchema
    conversation_trace: ConversationTraceDiagnosticsSchema
    # #500 3b-5: optional, only set on a non-legacy prompt_composition turn; `prompt` stays the
    # legacy joined string.
    composed_messages: list[ComposedMessageSchema] | None = None
    composition_report: CompositionReportSchema | None = None
    evidence_limit: dict[str, int | str | None] | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_composition_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Legacy turns must serialise exactly as before: drop ONLY these three keys when None
        # (other fields such as `loss` are legitimately null and stay).
        data: dict[str, Any] = handler(self)
        for name in ("composed_messages", "composition_report", "evidence_limit"):
            if data.get(name, 0) is None:
                del data[name]
        return data


class EmotionStateResponse(EmotionSchema):
    pass
