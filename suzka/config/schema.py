"""Typed configuration schema for PROJECT-SUZKA."""

import ipaddress
from pathlib import Path
import math
from urllib.parse import urlsplit

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from suzka.identifiers import validate_identifier
from suzka.limits import MAX_TRACE_ITEMS_PER_CONTEXT, MAX_TRACE_STABILITY_HOURS
from suzka.identity.value_system import ValueSeedDeclaration, ValueScope


def parse_approval_origin(origin: str) -> tuple[str, int]:
    """``http(s)://host[:port]`` -> (lowercase host, port); nothing else.

    No userinfo, path, query or fragment: an origin is a scheme, a host and an
    optional port (the default port of the scheme otherwise).
    """

    if type(origin) is not str or not origin.isascii() or len(origin) > 128:
        raise ValueError("an approval origin is a short ASCII http(s) origin")
    parts = urlsplit(origin)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path
        or parts.query
        or parts.fragment
        or origin != f"{parts.scheme}://{parts.netloc}"
        or origin != origin.lower()
    ):
        raise ValueError("an approval origin is http(s)://host[:port] only")
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError("an approval origin has an invalid port") from error
    host = parts.hostname
    return host, (port if port is not None else (443 if parts.scheme == "https" else 80))


def approval_host_header(origin: str) -> str:
    """The ``Host`` header value a browser sends for this origin (the port is
    omitted when it is the scheme's default)."""

    host, port = parse_approval_origin(origin)
    scheme = urlsplit(origin).scheme
    shown = f"[{host}]" if ":" in host else host
    default = 443 if scheme == "https" else 80
    return shown if port == default else f"{shown}:{port}"


def _is_loopback_ip(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_loopback_host(host: str) -> bool:
    """Whether a bind address is loopback (``localhost`` or a loopback IP)."""

    return host.strip().lower() == "localhost" or _is_loopback_ip(host.strip().strip("[]"))


def is_localhost_origin(origin: str) -> bool:
    """The direct-localhost transport's origins are exactly ``localhost``."""

    return parse_approval_origin(origin)[0] == "localhost"


class StrictBaseModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ProjectSettings(StrictBaseModel):
    name: str
    environment: str


class ModelSettings(StrictBaseModel):
    primary_id: str = Field(min_length=1)
    fallback_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    device: str
    dtype: str
    load_in_4bit: bool


class GenerationSettings(StrictBaseModel):
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    do_sample: bool


class EmotionSettings(StrictBaseModel):
    baseline_surprisal: float = Field(ge=0.0)
    high_emotion_threshold: float = Field(ge=0.0, le=1.0)
    decay_rate: float = Field(ge=0.0)
    timer_enabled: bool = False
    timer_interval_seconds: float = Field(default=60.0, gt=0.0)
    appraisal_response_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    resting_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    resting_arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    valence_recovery_rate: float = Field(default=0.01, ge=0.0)
    arousal_recovery_rate: float = Field(default=0.02, ge=0.0)

    @field_validator(
        "baseline_surprisal",
        "high_emotion_threshold",
        "decay_rate",
        "timer_interval_seconds",
        "appraisal_response_rate",
        "resting_valence",
        "resting_arousal",
        "valence_recovery_rate",
        "arousal_recovery_rate",
    )
    @classmethod
    def require_finite_emotion_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("emotion settings must be finite")
        return value


class AppraisalSettings(StrictBaseModel):
    initial_loss_scale: float = Field(default=1.0, gt=0.0)
    minimum_loss_scale: float = Field(default=0.01, gt=0.0)

    @field_validator("initial_loss_scale", "minimum_loss_scale")
    @classmethod
    def require_finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("loss calibration scales must be finite")
        return value


class MemorySettings(StrictBaseModel):
    persist_directory: Path
    db1_collection: str = Field(min_length=1)
    db2_collection: str = Field(min_length=1)
    db1_top_k: int = Field(gt=0)
    db2_top_k: int = Field(gt=0)
    embedding_model_id: str = Field(min_length=1)
    default_record_type: str = Field(min_length=1)


class WorkingMemorySettings(StrictBaseModel):
    item_capacity: int = Field(default=32, gt=0, le=4096)
    projection_max_bytes: int = Field(default=2048, gt=0, le=16 * 1024 * 1024)


class ConversationTraceSettings(StrictBaseModel):
    """#417 D1: the subject's short-term trace of recent Chat turns.

    Disabled by default. Enabling it moves AgentState to V11 one way; turning
    it off again afterwards freezes the trace instead (D1-b). The upper
    bounds are the schema constants in ``suzka.runtime.conversation_trace``.
    """

    enabled: StrictBool = False
    base_stability_hours: StrictFloat = Field(default=1.0, gt=0.0, le=168.0)
    surprise_gain: StrictFloat = Field(default=2.0, ge=0.0, le=10.0)
    recall_gain: StrictFloat = Field(default=0.5, ge=0.0, le=5.0)
    double_recall_gain: StrictFloat = Field(default=2.0, ge=0.0, le=10.0)
    max_stability_hours: StrictFloat = Field(
        default=168.0, gt=0.0, le=MAX_TRACE_STABILITY_HOURS
    )
    forget_below: StrictFloat = Field(default=0.2, gt=0.0, lt=1.0)
    verbatim_above: StrictFloat = Field(default=0.5, gt=0.0, lt=1.0)
    max_items_per_context: StrictInt = Field(
        default=20, ge=1, le=MAX_TRACE_ITEMS_PER_CONTEXT
    )
    # At least 1024 so a shortened CJK turn (~400 bytes) still fits (#421 review).
    max_rendered_bytes: StrictInt = Field(default=4096, ge=1024, le=16384)

    @model_validator(mode="after")
    def validate_relations(self) -> "ConversationTraceSettings":
        if self.forget_below >= self.verbatim_above:
            raise ValueError("forget_below must be below verbatim_above")
        if self.max_stability_hours < self.base_stability_hours:
            raise ValueError("max_stability_hours must be at least base_stability_hours")
        return self


class SleepSettings(StrictBaseModel):
    enabled: bool
    dream_dataset_path: Path
    min_emotion_score: float = Field(ge=0.0, le=1.0)
    max_episodes_per_cycle: int = Field(gt=0)


class QloraSettings(StrictBaseModel):
    output_dir: Path
    dry_run: bool
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    lora_dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    num_train_epochs: int = Field(gt=0)
    max_steps: int = Field(gt=0)


class AdapterRegistrySettings(StrictBaseModel):
    path: Path
    eval_result_dir: Path
    eval_sets: list[Path]
    trial_threshold: float = Field(ge=0.0, le=1.0)
    reject_threshold: float = Field(ge=0.0, le=1.0)
    allowed_states: list[str]
    manual_approval_required: bool


class ApiSettings(StrictBaseModel):
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    admin_token_env: str = Field(min_length=1)
    cors_origins: list[str]


class FrontendSettings(StrictBaseModel):
    base_url: str = Field(min_length=1)
    api_base_url: str = Field(min_length=1)


class RuntimeSettings(StrictBaseModel):
    queue_capacity: int = Field(default=64, gt=0)


class AgentStateSettings(StrictBaseModel):
    path: Path = Path(".suzka/agent_state.json")


class EventJournalSettings(StrictBaseModel):
    path: Path = Path(".suzka/event_journal.jsonl")
    max_bytes: int = Field(default=1_048_576, gt=0)
    # Deprecated (#381 D2 P2): retention is by time and compaction. Kept for
    # compatibility; it no longer deletes segments.
    retained_files: int = Field(default=4, ge=2)
    # Segments whose last record is older than this are archived (H2: ~30 days).
    retention_days: int = Field(default=30, ge=0, le=36_500)
    # Defaults to "<path>-archive" beside the Journal.
    archive_dir: Path | None = None
    # IP-4b (#381 §6): limits over live bytes (active + rotated segments;
    # archives are not counted). Above ``max_total_bytes - soft_band_bytes``
    # only cognition is admitted; above ``max_total_bytes`` no new event is
    # admitted. Defaults follow the #381 capacity table for ~100 turns/day
    # (72 MiB) with a soft band of more than 7 days of cognition (16 MiB).
    # ``max_total_bytes: null`` removes both limits; then soft_band_bytes
    # must be 0 as well.
    max_total_bytes: StrictInt | None = Field(default=75_497_472, gt=0)
    soft_band_bytes: StrictInt = Field(default=16_777_216, ge=0)

    @model_validator(mode="after")
    def validate_capacity_bands(self) -> "EventJournalSettings":
        if self.max_total_bytes is None:
            if self.soft_band_bytes != 0:
                raise ValueError("soft_band_bytes requires max_total_bytes")
        elif self.soft_band_bytes >= self.max_total_bytes:
            raise ValueError("soft_band_bytes must be below max_total_bytes")
        return self


class SelfGoalSettings(StrictBaseModel):
    """IP-11 (#379 D1 P0-1, D2 P0 / P2-3): the subject's own Goal proposal.

    Inert until the unique operator activation marker exists.  The values
    are the #379 proposals; ``min_strength`` 0.15 is salience >= 0.5 for an
    INTEREST (strength = 0.30 x salience x confidence, confidence 1.0).
    """

    min_strength: StrictFloat = Field(default=0.15, gt=0.0, le=1.0)
    per_incarnation: StrictInt = Field(default=3, ge=1, le=100)
    per_day: StrictInt = Field(default=3, ge=1, le=100)
    stale_after_hours: StrictInt = Field(default=24, ge=1, le=24 * 30)


class R18Settings(StrictBaseModel):
    """IP-14a (#411 §1, HP-6): governed Actions. Off until HP-6 says otherwise.

    While false, no R17 / R18 record (Intent, forecast, verdict, challenge,
    claim, Outcome, Assessment, Feedback) is written. A store that already
    holds such records starts switched off (IP-16b, #411 D1 r3 §6): it logs
    ``R18_RECORDS_PRESENT_BUT_DISABLED``, refuses every new record and keeps
    the Decision issuer's ledger reader running.
    """

    enabled: StrictBool = False
    # IP-16a (#411 D1 §3-5, Human decision §8-2): the fixed target and its
    # obligation secret (``obligation.key`` in the same private directory).
    # Only ``python -m suzka.r18.target_init`` creates them; startup never does.
    target_path: Path = Path(".suzka/private/r18/target.sqlite3")
    # IP-17a (#411 D1 r7 §3a, HD-16 (a)): each forward Intent's approval
    # window; an unclaimed Intent closes at its end with no event.
    approval_window_seconds: StrictInt = Field(default=3600, ge=600, le=86_400)
    # IP-17b (#411 D1 r7 §3 / §4): a challenge's TTL, capped by the Intent's
    # window end and by MAX_APPROVAL_TTL (15 min).
    approval_ttl_seconds: StrictInt = Field(default=300, ge=60, le=900)
    # IP-17c (#411 D1 r7 §4): the approval API's request guards.
    # ``approval_origins`` lists every Host name the Human uses (fail-closed):
    # ``http://localhost:<port>`` origins are the direct localhost transport;
    # any other origin (for example ``http://10.12.0.2:8080``, served by Caddy)
    # is a trusted-network origin and needs ``approval_trusted_networks``
    # (CIDRs; the networks the firewall already admits to Caddy).  An origin on
    # 127.0.0.1 / ::1 is refused (a relay's ``Host: 127.0.0.1:8000`` must never
    # match).  The default is localhost only.
    approval_origins: list[StrictStr] = Field(
        default_factory=lambda: ["http://localhost:8000"], min_length=1, max_length=16
    )
    approval_trusted_networks: list[StrictStr] = Field(default_factory=list, max_length=16)
    # IP-17c (§4): reviews per minute across the whole approval API.
    approval_reviews_per_minute: StrictInt = Field(default=10, ge=1, le=60)

    @field_validator("approval_origins")
    @classmethod
    def validate_approval_origins(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        for origin in value:
            host, _port = parse_approval_origin(origin)
            if _is_loopback_ip(host):
                raise ValueError("an approval origin must not be a loopback address")
            if origin in seen:
                raise ValueError("approval origins must be unique")
            seen.add(origin)
        return value

    @field_validator("approval_trusted_networks")
    @classmethod
    def validate_approval_networks(cls, value: list[str]) -> list[str]:
        for network in value:
            try:
                parsed = ipaddress.ip_network(network, strict=True)
            except ValueError as error:
                raise ValueError("approval_trusted_networks must be strict CIDRs") from error
            if parsed.prefixlen == 0:
                raise ValueError("a trusted network is never the whole address space")
        return value


class R19Settings(StrictBaseModel):
    """IP-14a (#411 §1, HP-6): Wakes, Scheduler and Outbox (IP-14b / IP-20)."""

    enabled: StrictBool = False


class SubjectHistorySettings(StrictBaseModel):
    """IP-9 U4-3 (#376 §A): the ``subject.history`` R07 participant.

    Enabling adopts a strictly larger R07 participant baseline at the next
    clean startup.  Adoption is one-way: once adopted, disabling the store
    makes startup fail closed rather than silently dropping a participant.
    """

    enabled: StrictBool = False
    path: Path = Path(".suzka/private/subject_history/subject_history.sqlite3")
    # SQLite page cap.  #376 D1-9 (Human-approved 2026-11-14): 65536 pages,
    # 256 MiB when the SQLite page size is the usual 4096 bytes.
    # IP-14a: an empty v2 store already takes 66 pages; IP-14b-1: v4, 99.
    max_page_count: StrictInt = Field(default=65_536, ge=128, le=2_147_483_646)
    # IP-14b D1 r2 HD-3 (a): PLAN_PREVIEW policy-HOLD receipts for the R20
    # report. Only lab_prepare turns it on for R20 subjects (IP-14b-3); eve
    # leaves it off and writes none. Independent of r18 / r19.
    plan_hold_receipts: StrictBool = False


class ChatLogSettings(StrictBaseModel):
    """Display-only Chat transcript for the Chat screen's sessions.

    It is never read by the runtime, so it does not enter prompts, memory,
    or learning. The default lets deployments without this section start.
    """

    path: Path = Path(".suzka/private/chat_log/chat_log.sqlite3")


class StateWALSettings(StrictBaseModel):
    directory: Path = Path(".suzka/private/state_wal")


class ValueSeedSettings(StrictBaseModel):
    """A non-authoritative value declaration supplied by configuration."""

    value_id: StrictStr
    name: StrictStr
    concept: StrictStr | None = None
    scope: Literal["subject", "context"] = "subject"
    context_ids: list[StrictStr] = Field(default_factory=list, max_length=16)
    polarity: StrictInt = 1
    strength: StrictFloat = Field(ge=0.0, le=1.0)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    stability: StrictFloat = Field(ge=0.0, le=1.0)
    protectedness: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    negotiability: StrictFloat = Field(default=1.0, ge=0.0, le=1.0)
    allowed_update_rate: StrictFloat = Field(gt=0.0, le=1.0)

    @field_validator("value_id")
    @classmethod
    def validate_value_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value) or len(value) > 128:
            raise ValueError("value text must be bounded and single-line")
        return value

    @field_validator("concept")
    @classmethod
    def validate_concept(cls, value: str | None) -> str | None:
        if value is not None and (
            not value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or len(value) > 2048
        ):
            raise ValueError("value text must be bounded and single-line")
        return value

    @field_validator("context_ids")
    @classmethod
    def validate_context_ids(cls, value: list[str]) -> list[str]:
        for context_id in value:
            validate_identifier(context_id)
        if value != sorted(set(value)) or len(value) != len(set(value)):
            raise ValueError("context_ids must be sorted and unique")
        return value

    @field_validator(
        "strength",
        "confidence",
        "stability",
        "protectedness",
        "negotiability",
        "allowed_update_rate",
    )
    @classmethod
    def require_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value settings must be finite")
        return value

    @field_validator("polarity")
    @classmethod
    def validate_polarity(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("polarity must be exactly -1 or 1")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> "ValueSeedSettings":
        if self.scope == "subject" and self.context_ids:
            raise ValueError("subject values cannot have contexts")
        if self.scope == "context" and not self.context_ids:
            raise ValueError("context values require contexts")
        return self

    def to_declaration(self) -> ValueSeedDeclaration:
        """Convert inert config into an immutable, digestable seed declaration."""

        return ValueSeedDeclaration(
            value_id=self.value_id,
            name=self.name,
            concept=self.concept,
            scope=ValueScope(self.scope),
            context_ids=tuple(self.context_ids),
            polarity=self.polarity,
            initial_strength=self.strength,
            confidence=self.confidence,
            stability=self.stability,
            protectedness=self.protectedness,
            negotiability=self.negotiability,
            allowed_update_rate=self.allowed_update_rate,
        )


class ValueConflictSettings(StrictBaseModel):
    conflict_id: StrictStr
    name: StrictStr
    left_value_id: StrictStr
    right_value_id: StrictStr

    @field_validator("conflict_id", "name")
    @classmethod
    def validate_conflict_text(cls, value: str) -> str:
        if not value or len(value) > 128 or any(
            ord(char) < 32 or ord(char) == 127 for char in value
        ):
            raise ValueError("conflict text must be bounded and single-line")
        return value

    @field_validator("conflict_id")
    @classmethod
    def validate_conflict_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("left_value_id", "right_value_id")
    @classmethod
    def validate_conflict_value_id(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def validate_pair_order(self) -> "ValueConflictSettings":
        if self.left_value_id == self.right_value_id:
            raise ValueError("conflicts cannot be self-conflicts")
        if self.left_value_id > self.right_value_id:
            raise ValueError("conflict value IDs must be in canonical order")
        return self


class ValueSystemSettings(StrictBaseModel):
    seeds: list[ValueSeedSettings] = Field(default_factory=list, max_length=128)
    conflicts: list[ValueConflictSettings] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def validate_references(self) -> "ValueSystemSettings":
        seed_ids = [seed.value_id for seed in self.seeds]
        present_ids = set(seed_ids)
        if len(present_ids) != len(seed_ids):
            raise ValueError("seed value IDs must be unique")
        pairs: set[tuple[str, str]] = set()
        conflict_ids: set[str] = set()
        for conflict in self.conflicts:
            pair = (conflict.left_value_id, conflict.right_value_id)
            if not {conflict.left_value_id, conflict.right_value_id} <= present_ids:
                raise ValueError("conflicts must reference known seed value IDs")
            if pair in pairs or (pair[1], pair[0]) in pairs:
                raise ValueError("conflict pairs must be unique")
            if conflict.conflict_id in conflict_ids:
                raise ValueError("conflict IDs must be unique")
            pairs.add(pair)
            conflict_ids.add(conflict.conflict_id)
        return self


class Settings(StrictBaseModel):
    project: ProjectSettings
    model: ModelSettings
    generation: GenerationSettings
    emotion: EmotionSettings
    appraisal: AppraisalSettings = Field(default_factory=AppraisalSettings)
    memory: MemorySettings
    sleep: SleepSettings
    qlora: QloraSettings
    adapter_registry: AdapterRegistrySettings
    api: ApiSettings
    frontend: FrontendSettings
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    working_memory: WorkingMemorySettings = Field(default_factory=WorkingMemorySettings)
    agent_state: AgentStateSettings = Field(default_factory=AgentStateSettings)
    event_journal: EventJournalSettings = Field(default_factory=EventJournalSettings)
    state_wal: StateWALSettings = Field(default_factory=StateWALSettings)
    chat_log: ChatLogSettings = Field(default_factory=ChatLogSettings)
    conversation_trace: ConversationTraceSettings = Field(
        default_factory=ConversationTraceSettings
    )
    subject_history: SubjectHistorySettings = Field(
        default_factory=SubjectHistorySettings
    )
    self_goal: SelfGoalSettings = Field(default_factory=SelfGoalSettings)
    r18: R18Settings = Field(default_factory=R18Settings)
    r19: R19Settings = Field(default_factory=R19Settings)
    values: ValueSystemSettings = Field(default_factory=ValueSystemSettings)

    @model_validator(mode="after")
    def validate_approval_binding(self) -> "Settings":
        require_approval_binding(self.api.host, self.r18)
        return self


def require_approval_binding(api_host: str, r18: R18Settings) -> None:
    """IP-17c (#411 D1 r7 §4 rule 1): with R18 on, the API must listen on
    loopback only, whatever the origins are, so every TCP peer is local
    (Caddy's, or a tunnel).  On a wider listen address any LAN host could send
    ``Host: localhost:<port>`` and pass the direct-localhost transport.  Called
    by the Settings validator and again at startup, because ``model_copy``
    skips validators."""

    if r18.enabled and not is_loopback_host(api_host):
        raise ValueError("r18.enabled needs api.host on loopback")
