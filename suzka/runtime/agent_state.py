"""Versioned, minimal, durable AgentState snapshot authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    cast,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from suzka.body import EmotionEngineAllostasis, EmotionState, EmotionTemporalState
from suzka.belief import (
    BELIEF_MAX_COMPONENT_CODEPOINTS,
    BELIEF_MAX_CONTEXTS,
    BELIEF_MAX_EVIDENCE,
    BELIEF_MAX_EVENT_SEQUENCE,
    BELIEF_MAX_PROPOSITION_CODEPOINTS,
    BELIEF_MAX_REVISIONS,
    BELIEF_MAX_RECORDS,
    BELIEF_MAX_REVISION,
    BELIEF_MAX_REVISION_WITNESSES,
    BELIEF_MAX_SERIALIZED_BYTES,
    BeliefEpistemicStatus,
    BeliefEvidenceType,
    BeliefLifecycle,
    BeliefSystem,
    BeliefRevisionOperation,
    BeliefRevisionReason,
    BeliefSubjectAdmissionReason,
    BeliefProposition,
    BeliefEvidence,
    BeliefRecord,
    BeliefRevisionRecord,
    BeliefSubjectAdmission,
    BeliefSystemSnapshot,
    belief_record_digest,
)
from suzka.limits import (
    MAX_CALIBRATION_SAMPLE_COUNT,
    MAX_PERSISTED_EVENT_SEQUENCE,
    MAX_PERSISTED_REVISION,
)
from suzka.cognition.surprisal_calculator import (
    CalibrationEntry,
    LossCalibration,
    _MAX_APPROVED_KEYS as CALIBRATION_MAX_ENTRIES,
)
from suzka.identity.origin import (
    IdentityOrigin,
    ORIGIN_MAX_EVENT_SEQUENCE,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
)
from suzka.identity.value_system import (
    VALUE_MAX_APPLIED_EVIDENCE_REFS,
    VALUE_MAX_AUTHORITATIVE_VALUES,
    VALUE_MAX_CONFLICTS,
    VALUE_MAX_CONCEPT_CODEPOINTS,
    VALUE_MAX_CONTEXT_IDS,
    VALUE_MAX_EVIDENCE_REFS,
    VALUE_MAX_EVENT_SEQUENCE,
    VALUE_MAX_NAME_CODEPOINTS,
    VALUE_MAX_OPPOSITION_COUNT,
    VALUE_MAX_REFS,
    VALUE_MAX_REVISION,
    VALUE_MAX_REVISION_RECORDS,
    ValueConflictDefinition,
    ValueDomainError,
    ValueRevisionHistory,
    ValueRevisionOperation,
    ValueRevisionRecord,
    ValueSeedDeclaration,
    ValueScope,
    ValueState,
    ValueSystem,
    recompute_seed_contract_digest,
    validate_value_history_immutable_basis,
    value_state_digest,
)
from suzka.privacy import normalize_private_key
from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS
from suzka.runtime.context import (
    ContextFrame,
    ContextRegistry,
    ContextRegistryState,
    ContextStatus,
    ContextType,
    InterlocutorBinding,
    MAX_CONTEXTS,
    MAX_EVIDENCE_REFERENCES,
    MAX_INTERLOCUTOR_BINDINGS,
    MAX_PARTICIPANTS_PER_CONTEXT,
    MAX_RELATIONS_PER_CONTEXT,
    validate_context_registry_state,
)
from suzka.runtime.working_memory import (
    MAX_ITEM_CAPACITY,
    WorkingMemory,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySourceKind,
    working_memory_item_id,
)

if TYPE_CHECKING:
    from suzka.runtime.main_loop import SuzkaMainLoop


AGENT_STATE_MAX_SERIALIZED_BYTES: Final[int] = 128 * 1024 * 1024
AGENT_STATE_FUTURE_STATE_RESERVE_BYTES: Final[int] = 16 * 1024 * 1024
_Identifier = Annotated[
    str,
    Field(min_length=1, max_length=MAX_IDENTIFIER_CODEPOINTS),
]
_Digest = Annotated[str, Field(min_length=64, max_length=64)]


@runtime_checkable
class BeliefStatePort(Protocol):
    """Runtime-owned persistence port; BeliefSystem remains authority owner.

    AgentState may serialize and restore through this port, but it does not
    receive mutation authority over Belief lifecycle, admission, or history.
    """

    def export_belief_state(self) -> BeliefSystemSnapshot: ...

    def restore_belief_state(self, snapshot: BeliefSystemSnapshot) -> None: ...


CURRENT_AGENT_STATE_SCHEMA_VERSION: Literal[7] = 7


class _StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EmotionStateSnapshot(_StateModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    optimal_loss: float = Field(ge=0.0)

    @field_validator("valence", "arousal", "optimal_loss")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("emotion value must be finite")
        return value


class _AgentStateSnapshotBase(_StateModel):
    saved_at: datetime
    last_processed_event_sequence: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    emotion_state: EmotionStateSnapshot

    @field_validator("saved_at", mode="before")
    @classmethod
    def parse_saved_at(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("saved_at must be a valid datetime") from error
        return value

    @field_validator("saved_at")
    @classmethod
    def require_aware_saved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("saved_at must be timezone-aware")
        return value

    @field_validator("last_processed_event_sequence", mode="before")
    @classmethod
    def reject_boolean_sequence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("sequence must be an integer")
        return value


class AgentStateSnapshotV1(_AgentStateSnapshotBase):
    """Exact retained R04-R07 canonical AgentState schema."""

    schema_version: Literal[1] = 1


class WorkingMemoryItemSnapshot(_StateModel):
    """Strict durable form of one U1 authoritative reference item."""

    item_id: str
    source_kind: Literal["episodic", "semantic"]
    source_id: str
    activation: float = Field(ge=0.0, le=1.0)
    salience: float = Field(ge=0.0, le=1.0)
    retention_reason: Literal["recent", "reactivated"]
    created_revision: int = Field(ge=0, le=MAX_PERSISTED_REVISION)
    last_activated_revision: int = Field(ge=0, le=MAX_PERSISTED_REVISION)

    @field_validator("activation", "salience")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Working Memory value must be finite")
        return value

    @field_validator("created_revision", "last_activated_revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Working Memory revision must be an integer")
        return value

    @model_validator(mode="after")
    def require_canonical_reference(self) -> WorkingMemoryItemSnapshot:
        try:
            source_kind = WorkingMemorySourceKind(self.source_kind)
            expected = working_memory_item_id(source_kind, self.source_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Working Memory source reference is invalid") from error
        if self.item_id != expected:
            raise ValueError("Working Memory item identity is invalid")
        return self


class WorkingMemorySnapshot(_StateModel):
    """Canonical Working Memory authority embedded in AgentState v2."""

    revision: int = Field(ge=0, le=MAX_PERSISTED_REVISION)
    items: tuple[WorkingMemoryItemSnapshot, ...] = Field(
        max_length=MAX_ITEM_CAPACITY
    )

    @field_validator("items", mode="before")
    @classmethod
    def parse_json_items(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Working Memory revision must be an integer")
        return value

    @model_validator(mode="after")
    def require_consistent_membership(self) -> WorkingMemorySnapshot:
        item_ids: set[str] = set()
        source_references: set[tuple[str, str]] = set()
        for item in self.items:
            if (
                item.created_revision > self.revision
                or item.last_activated_revision > self.revision
            ):
                raise ValueError("Working Memory item revision is invalid")
            source_reference = (item.source_kind, item.source_id)
            if item.item_id in item_ids or source_reference in source_references:
                raise ValueError("Working Memory item is duplicated")
            item_ids.add(item.item_id)
            source_references.add(source_reference)
        return self


class AgentStateSnapshotV2(_AgentStateSnapshotBase):
    """Exact retained R08 canonical AgentState schema."""

    schema_version: Literal[2] = 2
    working_memory: WorkingMemorySnapshot


def _parse_context_timestamp(value: object) -> object:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Context timestamp must be a valid datetime") from error
    return value


def _require_context_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Context timestamp must be canonical UTC")
    return value.astimezone(timezone.utc)


def _tuple_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


class ContextFrameSnapshot(_StateModel):
    """Strict durable projection of one Context frame."""

    context_id: str
    context_type: Literal["conversation"]
    source_channel: str
    source_session_id: str | None
    participant_refs: tuple[str, ...]
    parent_context_id: str | None
    related_context_ids: tuple[str, ...]
    status: Literal["active", "suspended", "closed"]
    created_revision: int = Field(ge=0)
    last_modified_revision: int = Field(ge=0)
    started_at: datetime
    last_active_at: datetime

    @field_validator(
        "participant_refs", "related_context_ids", mode="before"
    )
    @classmethod
    def parse_reference_lists(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("created_revision", "last_modified_revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Context revision must be an integer")
        return value

    @field_validator("started_at", "last_active_at", mode="before")
    @classmethod
    def parse_timestamps(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("started_at", "last_active_at")
    @classmethod
    def require_utc_timestamps(cls, value: datetime) -> datetime:
        return _require_context_utc(value)


class InterlocutorBindingSnapshot(_StateModel):
    """Strict durable projection of one evidence-bound interlocutor binding."""

    reference_key: str
    identity_key: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_references: tuple[str, ...]
    created_revision: int = Field(ge=0)
    last_modified_revision: int = Field(ge=0)

    @field_validator("evidence_references", mode="before")
    @classmethod
    def parse_reference_list(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Context confidence must be finite")
        return value

    @field_validator("created_revision", "last_modified_revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Context revision must be an integer")
        return value


class ContextStateSnapshot(_StateModel):
    """Canonical Context authority embedded in AgentState v3."""

    revision: int = Field(ge=0)
    current_context_id: str | None
    frames: tuple[ContextFrameSnapshot, ...]
    interlocutor_bindings: tuple[InterlocutorBindingSnapshot, ...]

    @field_validator("frames", "interlocutor_bindings", mode="before")
    @classmethod
    def parse_snapshot_lists(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("revision", mode="before")
    @classmethod
    def reject_boolean_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Context revision must be an integer")
        return value

    def to_registry_state(self) -> ContextRegistryState:
        return ContextRegistryState(
            revision=self.revision,
            current_context_id=self.current_context_id,
            frames=tuple(
                ContextFrame(
                    context_id=frame.context_id,
                    context_type=ContextType(frame.context_type),
                    source_channel=frame.source_channel,
                    source_session_id=frame.source_session_id,
                    participant_refs=frame.participant_refs,
                    parent_context_id=frame.parent_context_id,
                    related_context_ids=frame.related_context_ids,
                    status=ContextStatus(frame.status),
                    created_revision=frame.created_revision,
                    last_modified_revision=frame.last_modified_revision,
                    started_at=frame.started_at,
                    last_active_at=frame.last_active_at,
                )
                for frame in self.frames
            ),
            interlocutor_bindings=tuple(
                InterlocutorBinding(
                    reference_key=binding.reference_key,
                    identity_key=binding.identity_key,
                    confidence=binding.confidence,
                    evidence_references=binding.evidence_references,
                    created_revision=binding.created_revision,
                    last_modified_revision=binding.last_modified_revision,
                )
                for binding in self.interlocutor_bindings
            ),
        )

    @model_validator(mode="after")
    def require_valid_registry_state(self) -> ContextStateSnapshot:
        validate_context_registry_state(self.to_registry_state())
        return self


class AgentStateSnapshotV3(_AgentStateSnapshotBase):
    """Exact retained R09 v3 canonical snapshot."""

    schema_version: Literal[3] = 3
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


class CalibrationEntrySnapshot(_StateModel):
    """Opaque, exact persisted Welford calibration state."""

    model_key: str
    count: int = Field(ge=0)
    mean: float
    m2: float = Field(ge=0.0)

    @field_validator("model_key")
    @classmethod
    def require_opaque_key(cls, value: str) -> str:
        if len(value) != 70 or not value.startswith("model.") or any(
            character not in "0123456789abcdef" for character in value[6:]
        ):
            raise ValueError("model_key must be an opaque model key")
        return value

    @field_validator("count", mode="before")
    @classmethod
    def reject_boolean_count(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("count must be an integer")
        return value

    @field_validator("mean", "m2")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("calibration values must be finite")
        return value

    @model_validator(mode="after")
    def require_exact_statistics(self) -> CalibrationEntrySnapshot:
        # Use the authoritative U1 validator rather than duplicating its rules.
        CalibrationEntry(self.model_key, self.count, self.mean, self.m2)
        return self

    def to_entry(self) -> CalibrationEntry:
        return CalibrationEntry(self.model_key, self.count, self.mean, self.m2)


class AppraisalStateSnapshot(_StateModel):
    calibration_entries: tuple[CalibrationEntrySnapshot, ...]
    last_emotion_update_at: datetime | None

    @field_validator("calibration_entries", mode="before")
    @classmethod
    def parse_entries(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("last_emotion_update_at", mode="before")
    @classmethod
    def parse_timestamp(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("last_emotion_update_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_context_utc(value)

    @model_validator(mode="after")
    def require_ordered_unique_entries(self) -> AppraisalStateSnapshot:
        keys = tuple(entry.model_key for entry in self.calibration_entries)
        if len(keys) > 64:
            raise ValueError("calibration entries are bounded")
        if keys != tuple(sorted(set(keys))):
            raise ValueError("calibration entries must be ordered and unique")
        return self


class IdentityOriginSnapshot(_StateModel):
    """Strict durable projection of one Value provenance record."""

    actor: Literal[
        "self",
        "user",
        "operator",
        "system",
        "external_source",
        "model_inference",
        "inherited",
        "unknown",
    ]
    input_kind: Literal[
        "internal_state",
        "request",
        "suggestion",
        "constraint",
        "feedback",
        "evidence",
        "config_seed",
        "legacy",
    ]
    admission: Literal[
        "pending",
        "self_endorsed",
        "system_authorized",
        "rejected",
        "uncertain",
    ]
    source_ref: str | None
    event_id: str | None
    context_id: str | None
    event_sequence: int | None
    confidence: float
    origin_id: str

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Value origin confidence must be finite and bounded")
        return value

    @field_validator("event_sequence", mode="before")
    @classmethod
    def reject_boolean_event_sequence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value origin event_sequence must be an integer")
        return value


class ValueStateSnapshot(_StateModel):
    """Strict durable projection of one complete current Value state."""

    value_id: str
    revision: int = Field(ge=0)
    name: str
    concept: str | None
    scope: Literal["subject", "context"]
    context_ids: tuple[str, ...]
    polarity: int
    strength: float
    confidence: float
    stability: float
    protectedness: float
    negotiability: float
    allowed_update_rate: float
    frozen: bool
    origin: IdentityOriginSnapshot
    evidence_refs: tuple[str, ...]
    seed_contract_digest: str | None
    opposition_count: int = Field(ge=0, le=6)

    @field_validator("context_ids", "evidence_refs", mode="before")
    @classmethod
    def parse_value_lists(cls, value: object) -> object:
        return _tuple_value(value)

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
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Value scalar must be finite and bounded")
        return value

    @field_validator("revision", "opposition_count", mode="before")
    @classmethod
    def reject_boolean_value_integer(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value integer fields must be integers")
        return value


class ValueConflictSnapshot(_StateModel):
    left_value_id: str
    right_value_id: str


class ValueRevisionRecordSnapshot(_StateModel):
    """Strict durable projection of one immutable Value revision record."""

    value_id: str
    from_revision: int
    to_revision: int
    before_digest: str
    after_state_projection: ValueStateSnapshot
    after_digest: str
    operation: Literal[
        "admission", "update", "freeze", "unfreeze", "rollback", "origin_review"
    ]
    origin_id: str
    evidence_refs: tuple[str, ...]
    event_id: str
    event_sequence: int
    recorded_at: datetime
    previous_record_digest: str | None
    target_revision: int | None
    record_digest: str

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def parse_record_refs(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("recorded_at", mode="before")
    @classmethod
    def parse_record_timestamp(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("recorded_at")
    @classmethod
    def require_record_utc(cls, value: datetime) -> datetime:
        return _require_context_utc(value)

    @field_validator(
        "from_revision", "to_revision", "event_sequence", "target_revision", mode="before"
    )
    @classmethod
    def reject_boolean_record_integer(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value revision fields must be integers")
        return value


class ValueRevisionHistorySnapshot(_StateModel):
    value_id: str
    history_anchor_revision: int | None
    history_anchor_digest: str | None
    history_anchor_state_digest: str | None
    records: tuple[ValueRevisionRecordSnapshot, ...]

    @field_validator("records", mode="before")
    @classmethod
    def parse_history_records(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("history_anchor_revision", mode="before")
    @classmethod
    def reject_boolean_anchor_revision(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("history_anchor_revision must be an integer")
        return value


class ValueEvidenceLedgerSnapshot(_StateModel):
    value_id: str
    evidence_refs: tuple[str, ...]
    ledger_digest: str

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def parse_ledger_refs(cls, value: object) -> object:
        return _tuple_value(value)


class ValueSystemStateSnapshot(_StateModel):
    """Complete durable Value authority embedded in AgentState v5."""

    schema_version: Literal[1] = 1
    values: tuple[ValueStateSnapshot, ...]
    conflicts: tuple[ValueConflictSnapshot, ...]
    histories: tuple[ValueRevisionHistorySnapshot, ...]
    evidence_ledgers: tuple[ValueEvidenceLedgerSnapshot, ...]

    @field_validator("values", "conflicts", "histories", "evidence_ledgers", mode="before")
    @classmethod
    def parse_value_snapshot_lists(cls, value: object) -> object:
        return _tuple_value(value)

    @model_validator(mode="after")
    def require_canonical_order(self) -> ValueSystemStateSnapshot:
        value_ids = tuple(value.value_id for value in self.values)
        if value_ids != tuple(sorted(set(value_ids))):
            raise ValueError("Value snapshots must be ordered and unique")

        history_ids = tuple(history.value_id for history in self.histories)
        if history_ids != tuple(sorted(set(history_ids))):
            raise ValueError("Value histories must be ordered and unique")

        ledger_ids = tuple(ledger.value_id for ledger in self.evidence_ledgers)
        if ledger_ids != tuple(sorted(set(ledger_ids))):
            raise ValueError("Value evidence ledgers must be ordered and unique")

        conflict_pairs = tuple(
            (conflict.left_value_id, conflict.right_value_id)
            for conflict in self.conflicts
        )
        if any(left >= right for left, right in conflict_pairs):
            raise ValueError("Value conflict pairs must use canonical order")
        if conflict_pairs != tuple(sorted(set(conflict_pairs))):
            raise ValueError("Value conflict pairs must be ordered and unique")
        return self


class ValueRevisionDeltaSnapshot(_StateModel):
    """Compact mutable projection of one Value after-state.

    The immutable Value basis is owned by the corresponding current Value
    snapshot.  A revision retains only the fields which the existing Value
    authority may change, plus indexes into that Value's complete evidence
    ledger.  This is a persistence representation; it does not describe a
    domain mutation operation.
    """

    polarity: int
    strength: float
    confidence: float
    frozen: bool
    opposition_count: int
    origin_admission: Literal[
        "pending",
        "self_endorsed",
        "system_authorized",
        "rejected",
        "uncertain",
    ]
    evidence_ref_indices: tuple[int, ...]

    @field_validator("evidence_ref_indices", mode="before")
    @classmethod
    def parse_evidence_indices(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("evidence_ref_indices")
    @classmethod
    def require_canonical_evidence_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) > VALUE_MAX_EVIDENCE_REFS:
            raise ValueError("Value revision evidence indexes exceed their bound")
        if any(
            type(index) is not int
            or not 0 <= index < VALUE_MAX_APPLIED_EVIDENCE_REFS
            for index in value
        ):
            raise ValueError("Value revision evidence index is outside its bound")
        if value != tuple(sorted(set(value))):
            raise ValueError("Value revision evidence indexes must be ordered and unique")
        return value

    @field_validator("polarity", "opposition_count", mode="before")
    @classmethod
    def reject_boolean_delta_integer(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value revision delta integer fields must be integers")
        return value

    @field_validator("polarity")
    @classmethod
    def require_delta_polarity(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("Value revision delta polarity must be -1 or 1")
        return value

    @field_validator("strength", "confidence")
    @classmethod
    def require_delta_fraction(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Value revision delta scalar must be finite and bounded")
        return value

    @field_validator("opposition_count")
    @classmethod
    def require_delta_opposition_count(cls, value: int) -> int:
        if not 0 <= value <= VALUE_MAX_OPPOSITION_COUNT:
            raise ValueError("Value revision delta opposition count is outside its bound")
        return value


class ValueRevisionRecordSnapshotV7(_StateModel):
    """Normalized durable projection of one immutable Value revision record."""

    value_id: _Identifier
    from_revision: int = Field(ge=-1, le=VALUE_MAX_REVISION)
    to_revision: int = Field(ge=0, le=VALUE_MAX_REVISION)
    before_digest: _Digest
    after_state_delta: ValueRevisionDeltaSnapshot
    after_digest: _Digest
    operation: Literal[
        "admission", "update", "freeze", "unfreeze", "rollback", "origin_review"
    ]
    origin_id: _Digest
    evidence_ref_indices: tuple[int, ...]
    event_id: _Identifier
    event_sequence: int
    recorded_at: datetime
    previous_record_digest: _Digest | None
    target_revision: int | None
    record_digest: _Digest

    @field_validator("evidence_ref_indices", mode="before")
    @classmethod
    def parse_record_evidence_indices(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("evidence_ref_indices")
    @classmethod
    def require_record_evidence_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) > VALUE_MAX_REFS:
            raise ValueError("Value revision record evidence indexes exceed their bound")
        if any(
            type(index) is not int
            or not 0 <= index < VALUE_MAX_APPLIED_EVIDENCE_REFS
            for index in value
        ):
            raise ValueError("Value revision record evidence index is outside its bound")
        if value != tuple(sorted(set(value))):
            raise ValueError("Value revision record evidence indexes must be ordered and unique")
        return value

    @field_validator("recorded_at", mode="before")
    @classmethod
    def parse_record_timestamp_v7(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("recorded_at")
    @classmethod
    def require_record_utc_v7(cls, value: datetime) -> datetime:
        return _require_context_utc(value)

    @field_validator(
        "from_revision", "to_revision", "event_sequence", "target_revision", mode="before"
    )
    @classmethod
    def reject_boolean_record_integer_v7(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Value revision fields must be integers")
        return value

    @field_validator("from_revision")
    @classmethod
    def require_from_revision_bound(cls, value: int) -> int:
        if not -1 <= value <= VALUE_MAX_REVISION:
            raise ValueError("Value from_revision is outside its bound")
        return value

    @field_validator("to_revision", "target_revision")
    @classmethod
    def require_revision_bound(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= VALUE_MAX_REVISION:
            raise ValueError("Value revision is outside its bound")
        return value

    @field_validator("event_sequence")
    @classmethod
    def require_event_sequence_bound(cls, value: int) -> int:
        if not 0 <= value <= VALUE_MAX_EVENT_SEQUENCE:
            raise ValueError("Value event sequence is outside its bound")
        return value


class ValueRevisionHistorySnapshotV7(_StateModel):
    """A complete retained Value history using normalized revision records."""

    value_id: _Identifier
    history_anchor_revision: int | None = Field(
        default=None, ge=0, le=VALUE_MAX_REVISION
    )
    history_anchor_digest: _Digest | None
    history_anchor_state_digest: _Digest | None
    records: tuple[ValueRevisionRecordSnapshotV7, ...]

    @field_validator("records", mode="before")
    @classmethod
    def parse_history_records_v7(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("history_anchor_revision", mode="before")
    @classmethod
    def reject_boolean_anchor_revision_v7(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("history_anchor_revision must be an integer")
        return value

    @field_validator("history_anchor_revision")
    @classmethod
    def require_anchor_revision_bound_v7(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= VALUE_MAX_REVISION:
            raise ValueError("history_anchor_revision is outside its bound")
        return value

    @model_validator(mode="after")
    def require_bounded_history_v7(self) -> ValueRevisionHistorySnapshotV7:
        if len(self.records) > VALUE_MAX_REVISION_RECORDS:
            raise ValueError("Value revision history exceeds its retained record bound")
        anchor_values = (
            self.history_anchor_revision,
            self.history_anchor_digest,
            self.history_anchor_state_digest,
        )
        if any(value is None for value in anchor_values) and not all(
            value is None for value in anchor_values
        ):
            raise ValueError("Value history anchor fields must be all present or absent")
        return self


class ValueSystemStateSnapshotV7(_StateModel):
    """Complete durable Value state with normalized retained histories."""

    schema_version: Literal[2] = 2
    values: tuple[ValueStateSnapshot, ...] = Field(
        max_length=VALUE_MAX_AUTHORITATIVE_VALUES
    )
    conflicts: tuple[ValueConflictSnapshot, ...] = Field(
        max_length=VALUE_MAX_CONFLICTS
    )
    histories: tuple[ValueRevisionHistorySnapshotV7, ...] = Field(
        max_length=VALUE_MAX_AUTHORITATIVE_VALUES
    )
    evidence_ledgers: tuple[ValueEvidenceLedgerSnapshot, ...] = Field(
        max_length=VALUE_MAX_AUTHORITATIVE_VALUES
    )

    @field_validator("values", "conflicts", "histories", "evidence_ledgers", mode="before")
    @classmethod
    def parse_value_snapshot_lists_v7(cls, value: object) -> object:
        return _tuple_value(value)

    @model_validator(mode="after")
    def require_canonical_order_v7(self) -> ValueSystemStateSnapshotV7:
        value_ids = tuple(value.value_id for value in self.values)
        if len(value_ids) > VALUE_MAX_AUTHORITATIVE_VALUES:
            raise ValueError("Value snapshots exceed their authoritative bound")
        if value_ids != tuple(sorted(set(value_ids))):
            raise ValueError("Value snapshots must be ordered and unique")

        history_ids = tuple(history.value_id for history in self.histories)
        if len(history_ids) > VALUE_MAX_AUTHORITATIVE_VALUES:
            raise ValueError("Value histories exceed their authoritative bound")
        if history_ids != tuple(sorted(set(history_ids))):
            raise ValueError("Value histories must be ordered and unique")

        ledger_ids = tuple(ledger.value_id for ledger in self.evidence_ledgers)
        if len(ledger_ids) > VALUE_MAX_AUTHORITATIVE_VALUES:
            raise ValueError("Value evidence ledgers exceed their authoritative bound")
        if ledger_ids != tuple(sorted(set(ledger_ids))):
            raise ValueError("Value evidence ledgers must be ordered and unique")

        conflict_pairs = tuple(
            (conflict.left_value_id, conflict.right_value_id)
            for conflict in self.conflicts
        )
        max_conflicts = VALUE_MAX_AUTHORITATIVE_VALUES * (
            VALUE_MAX_AUTHORITATIVE_VALUES - 1
        ) // 2
        if len(conflict_pairs) > max_conflicts:
            raise ValueError("Value conflicts exceed their authoritative bound")
        if conflict_pairs != tuple(sorted(set(conflict_pairs))):
            raise ValueError("Value conflict pairs must be ordered and unique")
        if any(left >= right for left, right in conflict_pairs):
            raise ValueError("Value conflict pairs must use canonical order")
        return self


# Descriptive aliases keep the normalized codec discoverable without creating
# parallel schemas.  The persisted class names above are the canonical API.
ValueStateDeltaSnapshot = ValueRevisionDeltaSnapshot
ValueRevisionRecordV7Snapshot = ValueRevisionRecordSnapshotV7
ValueRevisionHistoryV7Snapshot = ValueRevisionHistorySnapshotV7


class BeliefPropositionStateSnapshot(_StateModel):
    canonical_text: str = Field(max_length=BELIEF_MAX_PROPOSITION_CODEPOINTS)
    subject: str | None = Field(default=None, max_length=BELIEF_MAX_COMPONENT_CODEPOINTS)
    predicate: str | None = Field(default=None, max_length=BELIEF_MAX_COMPONENT_CODEPOINTS)
    object: str | None = Field(default=None, max_length=BELIEF_MAX_COMPONENT_CODEPOINTS)
    proposition_digest: str = Field(min_length=64, max_length=64)


class BeliefEvidenceStateSnapshot(_StateModel):
    evidence_ref: str = Field(min_length=1, max_length=128)
    evidence_type: Literal[
        "experience",
        "semantic_memory",
        "episodic_memory",
        "external_claim",
        "model_inference",
        "operator_correction",
    ]


class BeliefSubjectAdmissionStateSnapshot(_StateModel):
    proposition_digest: str = Field(min_length=64, max_length=64)
    evidence_refs: tuple[str, ...]
    event_id: str = Field(min_length=1, max_length=128)
    event_sequence: int = Field(ge=1)
    reason: Literal["subject_endorsement", "subject_correction", "subject_review"]
    admission_digest: str = Field(min_length=64, max_length=64)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def parse_admission_refs(cls, value: object) -> object:
        return _tuple_value(value)

    @model_validator(mode="after")
    def require_canonical_refs(self) -> BeliefSubjectAdmissionStateSnapshot:
        if len(self.evidence_refs) == 0 or len(self.evidence_refs) > BELIEF_MAX_EVIDENCE:
            raise ValueError("Belief admission evidence is outside its bound")
        if self.evidence_refs != tuple(sorted(set(self.evidence_refs))):
            raise ValueError("Belief admission evidence must be ordered and unique")
        return self


class BeliefRevisionStateSnapshot(_StateModel):
    belief_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    operation: Literal["create", "adopt", "correct", "supersede", "retract", "expire"]
    reason: Literal[
        "creation",
        "subject_admission",
        "correction",
        "supersession",
        "retraction",
        "expiration",
    ]
    created_at: datetime
    previous_revision_digest: str | None = Field(default=None, min_length=64, max_length=64)
    event_id: str = Field(min_length=1, max_length=128)
    event_sequence: int = Field(ge=1)
    evidence_refs: tuple[str, ...]
    record_digest: str = Field(min_length=64, max_length=64)

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_revision_timestamp(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("created_at")
    @classmethod
    def require_revision_utc(cls, value: datetime) -> datetime:
        return _require_context_utc(value)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def parse_revision_refs(cls, value: object) -> object:
        return _tuple_value(value)

    @model_validator(mode="after")
    def require_canonical_refs(self) -> BeliefRevisionStateSnapshot:
        if len(self.evidence_refs) > BELIEF_MAX_REVISION_WITNESSES:
            raise ValueError("Belief revision witnesses exceed their bound")
        if self.evidence_refs != tuple(sorted(set(self.evidence_refs))):
            raise ValueError("Belief revision evidence must be ordered and unique")
        return self


class BeliefRecordStateSnapshot(_StateModel):
    belief_id: str = Field(min_length=1, max_length=128)
    proposition: BeliefPropositionStateSnapshot
    lifecycle: Literal["proposed", "adopted", "superseded", "retracted", "expired"]
    epistemic_status: Literal["unknown", "uncertain", "probable", "established"]
    confidence: float = Field(ge=0.0, le=1.0)
    context_scope: tuple[str, ...]
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    evidence: tuple[BeliefEvidenceStateSnapshot, ...]
    subject_admission: BeliefSubjectAdmissionStateSnapshot | None = None
    supersedes_id: str | None = Field(default=None, max_length=128)
    superseded_by_id: str | None = Field(default=None, max_length=128)
    revision: int = Field(ge=0)
    revision_history: tuple[BeliefRevisionStateSnapshot, ...]
    history_anchor_digest: str | None = Field(default=None, min_length=64, max_length=64)
    schema_version: Literal[1] = 1
    record_digest: str = Field(min_length=64, max_length=64)

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Belief confidence must be finite")
        return value

    @field_validator(
        "context_scope", "evidence", "revision_history", mode="before"
    )
    @classmethod
    def parse_belief_lists(cls, value: object) -> object:
        return _tuple_value(value)

    @field_validator("valid_from", "valid_until", mode="before")
    @classmethod
    def parse_belief_timestamps(cls, value: object) -> object:
        return _parse_context_timestamp(value)

    @field_validator("valid_from", "valid_until")
    @classmethod
    def require_belief_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_context_utc(value)

    @model_validator(mode="after")
    def require_canonical_collections(self) -> BeliefRecordStateSnapshot:
        if len(self.context_scope) > BELIEF_MAX_CONTEXTS:
            raise ValueError("Belief Context scope exceeds its bound")
        if self.context_scope != tuple(sorted(set(self.context_scope))):
            raise ValueError("Belief Context scope must be ordered and unique")
        if len(self.evidence) > BELIEF_MAX_EVIDENCE:
            raise ValueError("Belief evidence exceeds its bound")
        evidence_refs = tuple(item.evidence_ref for item in self.evidence)
        if evidence_refs != tuple(sorted(set(evidence_refs))):
            raise ValueError("Belief evidence must be ordered and unique")
        if len(self.revision_history) > BELIEF_MAX_REVISIONS:
            raise ValueError("Belief revision history exceeds its bound")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


_BELIEF_MAX_DATETIME_JSON = "9999-12-31T23:59:59.999999Z"
_BELIEF_MAX_IDENTIFIER = "a" * MAX_IDENTIFIER_CODEPOINTS
_BELIEF_MAX_BELIEF_ID = "b" * len("belief-" + "0" * 64)
_BELIEF_MAX_DIGEST = "c" * 64
_BELIEF_SCHEMA_MAX_EVENT_SEQUENCE = BELIEF_MAX_EVENT_SEQUENCE
_BELIEF_SCHEMA_MAX_REVISION = BELIEF_MAX_REVISION
# D7 reserves the longest JSON representation of a finite non-negative
# binary64 confidence value accepted by the domain.  The current CPython
# encoder emits at most 23 bytes for this range; one byte of slack keeps this
# schema budget independent of a shorter future formatter.
BELIEF_SCHEMA_MAX_FLOAT_JSON_BYTES: Final[int] = 24
_BELIEF_SCHEMA_MAX_CONFIDENCE = 1.2345678901234567e-300


def _schema_max_text(codepoints: int) -> str:
    """Return a valid worst-case UTF-8 text sample for a code-point bound."""

    return "\U00010000" * codepoints


def _schema_max_enum_value(enum_type: type[Enum]) -> str:
    return max((member.value for member in enum_type), key=len)


def _schema_max_belief_revision() -> dict[str, object]:
    return {
        "belief_id": _BELIEF_MAX_BELIEF_ID,
        "created_at": _BELIEF_MAX_DATETIME_JSON,
        "event_id": _BELIEF_MAX_IDENTIFIER,
        "event_sequence": _BELIEF_SCHEMA_MAX_EVENT_SEQUENCE,
        "evidence_refs": [_BELIEF_MAX_IDENTIFIER] * BELIEF_MAX_REVISION_WITNESSES,
        "operation": _schema_max_enum_value(BeliefRevisionOperation),
        "previous_revision_digest": _BELIEF_MAX_DIGEST,
        "reason": _schema_max_enum_value(BeliefRevisionReason),
        "record_digest": _BELIEF_MAX_DIGEST,
        "revision": _BELIEF_SCHEMA_MAX_REVISION,
    }


def _schema_max_belief_record() -> dict[str, object]:
    return {
        "belief_id": _BELIEF_MAX_BELIEF_ID,
        "confidence": _BELIEF_SCHEMA_MAX_CONFIDENCE,
        "context_scope": [_BELIEF_MAX_IDENTIFIER] * BELIEF_MAX_CONTEXTS,
        "epistemic_status": _schema_max_enum_value(BeliefEpistemicStatus),
        "evidence": [
            {
                "evidence_ref": _BELIEF_MAX_IDENTIFIER,
                "evidence_type": _schema_max_enum_value(BeliefEvidenceType),
            }
        ]
        * BELIEF_MAX_EVIDENCE,
        "history_anchor_digest": _BELIEF_MAX_DIGEST,
        "lifecycle": _schema_max_enum_value(BeliefLifecycle),
        "proposition": {
            "canonical_text": _schema_max_text(BELIEF_MAX_PROPOSITION_CODEPOINTS),
            "object": _schema_max_text(BELIEF_MAX_COMPONENT_CODEPOINTS),
            "predicate": _schema_max_text(BELIEF_MAX_COMPONENT_CODEPOINTS),
            "proposition_digest": _BELIEF_MAX_DIGEST,
            "subject": _schema_max_text(BELIEF_MAX_COMPONENT_CODEPOINTS),
        },
        "record_digest": _BELIEF_MAX_DIGEST,
        "revision": _BELIEF_SCHEMA_MAX_REVISION,
        "revision_history": [
            _schema_max_belief_revision()
        ]
        * BELIEF_MAX_REVISIONS,
        "schema_version": 1,
        "subject_admission": {
            "admission_digest": _BELIEF_MAX_DIGEST,
            "event_id": _BELIEF_MAX_IDENTIFIER,
            "event_sequence": _BELIEF_SCHEMA_MAX_EVENT_SEQUENCE,
            "evidence_refs": [_BELIEF_MAX_IDENTIFIER] * BELIEF_MAX_EVIDENCE,
            "proposition_digest": _BELIEF_MAX_DIGEST,
            "reason": _schema_max_enum_value(BeliefSubjectAdmissionReason),
        },
        "superseded_by_id": _BELIEF_MAX_BELIEF_ID,
        "supersedes_id": _BELIEF_MAX_BELIEF_ID,
        "valid_from": _BELIEF_MAX_DATETIME_JSON,
        "valid_until": _BELIEF_MAX_DATETIME_JSON,
    }


_BELIEF_SCHEMA_MAX_RECORD = _schema_max_belief_record()
_BELIEF_SCHEMA_CONFIDENCE_JSON_BYTES = len(
    json.dumps(
        _BELIEF_SCHEMA_MAX_CONFIDENCE,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
)
BELIEF_SCHEMA_MAX_RECORD_SERIALIZED_BYTES: Final[int] = len(
    _canonical_json_bytes(_BELIEF_SCHEMA_MAX_RECORD)
) + (BELIEF_SCHEMA_MAX_FLOAT_JSON_BYTES - _BELIEF_SCHEMA_CONFIDENCE_JSON_BYTES)
_BELIEF_SCHEMA_EMPTY_SECTION_BYTES = len(
    _canonical_json_bytes(
        {"authority_digest": _BELIEF_MAX_DIGEST, "records": [], "schema_version": 1}
    )
)
BELIEF_SCHEMA_MAX_SERIALIZED_BYTES: Final[int] = (
    _BELIEF_SCHEMA_EMPTY_SECTION_BYTES
    + BELIEF_MAX_RECORDS * (BELIEF_SCHEMA_MAX_RECORD_SERIALIZED_BYTES + 1)
    - 1
)


class BeliefSystemStateSnapshot(_StateModel):
    """Complete bounded intrinsic Belief authority embedded in AgentState v6.

    ``BELIEF_SCHEMA_MAX_SERIALIZED_BYTES`` is a schema-derived upper bound for
    this section; ``BELIEF_MAX_SERIALIZED_BYTES`` remains the runtime admission
    cap and is intentionally larger than the derived maximum.
    """

    schema_version: Literal[1] = 1
    records: tuple[BeliefRecordStateSnapshot, ...]
    authority_digest: str = Field(min_length=64, max_length=64)

    @field_validator("records", mode="before")
    @classmethod
    def parse_belief_records(cls, value: object) -> object:
        return _tuple_value(value)

    @model_validator(mode="after")
    def require_canonical_records(self) -> BeliefSystemStateSnapshot:
        if len(self.records) > BELIEF_MAX_RECORDS:
            raise ValueError("Belief authority exceeds its record bound")
        ids = tuple(record.belief_id for record in self.records)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Belief records must be ordered and unique")
        if len(_canonical_json_bytes(self.model_dump(mode="json"))) > BELIEF_MAX_SERIALIZED_BYTES:
            raise ValueError("Belief section exceeds its serialized byte bound")
        return self

class AgentStateSnapshotV4(_AgentStateSnapshotBase):
    """Exact retained R10 canonical AgentState v4 schema."""

    schema_version: Literal[4] = 4
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot
    appraisal_state: AppraisalStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


class AgentStateSnapshotV5(_AgentStateSnapshotBase):
    """Current AgentState v5 with complete Value authority continuity."""

    schema_version: Literal[5] = 5
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot
    appraisal_state: AppraisalStateSnapshot
    value_state: ValueSystemStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


class AgentStateSnapshotV6(_AgentStateSnapshotBase):
    """Current AgentState v6 with complete Belief and Value authority continuity."""

    schema_version: Literal[6] = 6
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot
    appraisal_state: AppraisalStateSnapshot
    value_state: ValueSystemStateSnapshot
    belief_state: BeliefSystemStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


class AgentStateSnapshotV7(_AgentStateSnapshotBase):
    """Current AgentState schema with normalized Value history storage."""

    schema_version: Literal[7] = CURRENT_AGENT_STATE_SCHEMA_VERSION
    working_memory: WorkingMemorySnapshot
    context_state: ContextStateSnapshot
    appraisal_state: AppraisalStateSnapshot
    value_state: ValueSystemStateSnapshotV7
    belief_state: BeliefSystemStateSnapshot

    @field_validator("saved_at")
    @classmethod
    def require_canonical_saved_at(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be canonical UTC")
        return value.astimezone(timezone.utc)


# The unqualified name denotes the current schema; retained callers should use
# AgentStateSnapshotV4 when they intentionally construct the exact v4 shape.
AgentStateSnapshot = AgentStateSnapshotV7


CompatibleAgentStateSnapshot = Annotated[
    AgentStateSnapshotV1
    | AgentStateSnapshotV2
    | AgentStateSnapshotV3
    | AgentStateSnapshotV4
    | AgentStateSnapshotV5
    | AgentStateSnapshotV6
    | AgentStateSnapshotV7,
    Field(discriminator="schema_version"),
]
_COMPATIBLE_SNAPSHOT_ADAPTER: TypeAdapter[CompatibleAgentStateSnapshot] = (
    TypeAdapter(CompatibleAgentStateSnapshot)
)


def validate_compatible_agent_state_snapshot(
    value: object,
) -> CompatibleAgentStateSnapshot:
    """Validate a compatible snapshot without changing its schema version."""

    return _COMPATIBLE_SNAPSHOT_ADAPTER.validate_python(value)


AGENT_STATE_V7_SCHEMA_MAX_FLOAT_JSON_BYTES: Final[int] = 25
# Retain the earlier descriptive name for callers which used the capacity
# helper before the normalized Value codec was introduced.  One byte of slack
# is reserved beyond the longest currently emitted finite binary64 literal.
AGENT_STATE_V6_SCHEMA_MAX_FLOAT_JSON_BYTES: Final[int] = (
    AGENT_STATE_V7_SCHEMA_MAX_FLOAT_JSON_BYTES
)
_SCHEMA_MAX_IDENTIFIER = "a" * MAX_IDENTIFIER_CODEPOINTS
_SCHEMA_MAX_DIGEST = "f" * 64
_SCHEMA_MAX_DATETIME = "9999-12-31T23:59:59.999999Z"
_SCHEMA_MAX_UNIT_FLOAT = 1.2345678901234567e-300
_SCHEMA_MAX_SIGNED_UNIT_FLOAT = -1.2345678901234567e-300
_SCHEMA_MAX_NONNEGATIVE_FLOAT = 1.7976931348623157e308
_SCHEMA_MAX_SIGNED_FLOAT = -1.7976931348623157e308


def _schema_max_json_bytes(value: object) -> int:
    """Return a bounded JSON byte maximum for one owner-shaped value."""

    encoded = _canonical_json_bytes(value)
    float_slack = sum(
        AGENT_STATE_V7_SCHEMA_MAX_FLOAT_JSON_BYTES
        - len(_canonical_json_bytes(float_value))
        for float_value in _schema_float_values(value)
    )
    return len(encoded) + float_slack


def _schema_float_values(value: object) -> tuple[float, ...]:
    if type(value) is float:
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            float_value
            for child in value.values()
            for float_value in _schema_float_values(child)
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            float_value
            for child in value
            for float_value in _schema_float_values(child)
        )
    return ()


def _schema_max_working_memory_item() -> dict[str, object]:
    source_id = _SCHEMA_MAX_IDENTIFIER
    source_kind = WorkingMemorySourceKind.SEMANTIC
    return {
        "activation": _SCHEMA_MAX_UNIT_FLOAT,
        "created_revision": MAX_PERSISTED_REVISION,
        "item_id": working_memory_item_id(source_kind, source_id),
        "last_activated_revision": MAX_PERSISTED_REVISION,
        "retention_reason": _schema_max_enum_value(WorkingMemoryRetentionReason),
        "salience": _SCHEMA_MAX_UNIT_FLOAT,
        "source_id": source_id,
        "source_kind": source_kind.value,
    }


def _schema_max_working_memory() -> dict[str, object]:
    item = _schema_max_working_memory_item()
    return {
        "items": [item for _ in range(MAX_ITEM_CAPACITY)],
        "revision": MAX_PERSISTED_REVISION,
    }


def _schema_max_context_frame() -> dict[str, object]:
    return {
        "context_id": _SCHEMA_MAX_IDENTIFIER,
        "context_type": ContextType.CONVERSATION.value,
        "created_revision": MAX_PERSISTED_REVISION,
        "last_active_at": _SCHEMA_MAX_DATETIME,
        "last_modified_revision": MAX_PERSISTED_REVISION,
        "parent_context_id": _SCHEMA_MAX_IDENTIFIER,
        "participant_refs": [
            _SCHEMA_MAX_IDENTIFIER for _ in range(MAX_PARTICIPANTS_PER_CONTEXT)
        ],
        "related_context_ids": [
            _SCHEMA_MAX_IDENTIFIER for _ in range(MAX_RELATIONS_PER_CONTEXT)
        ],
        "source_channel": _SCHEMA_MAX_IDENTIFIER,
        "source_session_id": _SCHEMA_MAX_IDENTIFIER,
        "started_at": _SCHEMA_MAX_DATETIME,
        "status": _schema_max_enum_value(ContextStatus),
    }


def _schema_max_interlocutor_binding() -> dict[str, object]:
    return {
        "confidence": _SCHEMA_MAX_UNIT_FLOAT,
        "created_revision": MAX_PERSISTED_REVISION,
        "evidence_references": [
            _SCHEMA_MAX_IDENTIFIER for _ in range(MAX_EVIDENCE_REFERENCES)
        ],
        "identity_key": _SCHEMA_MAX_IDENTIFIER,
        "last_modified_revision": MAX_PERSISTED_REVISION,
        "reference_key": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_context() -> dict[str, object]:
    frame = _schema_max_context_frame()
    binding = _schema_max_interlocutor_binding()
    return {
        "current_context_id": _SCHEMA_MAX_IDENTIFIER,
        "frames": [frame for _ in range(MAX_CONTEXTS)],
        "interlocutor_bindings": [
            binding for _ in range(MAX_INTERLOCUTOR_BINDINGS)
        ],
        "revision": MAX_PERSISTED_REVISION,
    }


def _schema_max_calibration_entry() -> dict[str, object]:
    return {
        "count": MAX_CALIBRATION_SAMPLE_COUNT,
        "m2": _SCHEMA_MAX_NONNEGATIVE_FLOAT,
        "mean": _SCHEMA_MAX_SIGNED_FLOAT,
        "model_key": "model." + "f" * 64,
    }


def _schema_max_appraisal() -> dict[str, object]:
    entry = _schema_max_calibration_entry()
    return {
        "calibration_entries": [
            entry for _ in range(CALIBRATION_MAX_ENTRIES)
        ],
        "last_emotion_update_at": _SCHEMA_MAX_DATETIME,
    }


def _schema_max_emotion() -> dict[str, object]:
    return {
        "arousal": _SCHEMA_MAX_UNIT_FLOAT,
        "optimal_loss": _SCHEMA_MAX_NONNEGATIVE_FLOAT,
        "valence": _SCHEMA_MAX_SIGNED_UNIT_FLOAT,
    }


def _schema_max_origin() -> dict[str, object]:
    return {
        "actor": _schema_max_enum_value(OriginActor),
        "admission": _schema_max_enum_value(ValueAdmissionStatus),
        "confidence": _SCHEMA_MAX_UNIT_FLOAT,
        "context_id": _SCHEMA_MAX_IDENTIFIER,
        "event_id": _SCHEMA_MAX_IDENTIFIER,
        "event_sequence": ORIGIN_MAX_EVENT_SEQUENCE,
        "input_kind": _schema_max_enum_value(OriginInputKind),
        "origin_id": _SCHEMA_MAX_DIGEST,
        "source_ref": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_value_state() -> dict[str, object]:
    return {
        "allowed_update_rate": _SCHEMA_MAX_UNIT_FLOAT,
        "confidence": _SCHEMA_MAX_UNIT_FLOAT,
        "concept": _schema_max_text(VALUE_MAX_CONCEPT_CODEPOINTS),
        "context_ids": [
            _SCHEMA_MAX_IDENTIFIER for _ in range(VALUE_MAX_CONTEXT_IDS)
        ],
        "evidence_refs": [
            _SCHEMA_MAX_IDENTIFIER for _ in range(VALUE_MAX_EVIDENCE_REFS)
        ],
        "frozen": True,
        "name": _schema_max_text(VALUE_MAX_NAME_CODEPOINTS),
        "opposition_count": VALUE_MAX_OPPOSITION_COUNT,
        "origin": _schema_max_origin(),
        "polarity": -1,
        "protectedness": _SCHEMA_MAX_UNIT_FLOAT,
        "negotiability": _SCHEMA_MAX_UNIT_FLOAT,
        "revision": VALUE_MAX_REVISION,
        "scope": _schema_max_enum_value(ValueScope),
        "seed_contract_digest": _SCHEMA_MAX_DIGEST,
        "stability": _SCHEMA_MAX_UNIT_FLOAT,
        "strength": _SCHEMA_MAX_UNIT_FLOAT,
        "value_id": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_value_revision_delta() -> dict[str, object]:
    return {
        "confidence": _SCHEMA_MAX_UNIT_FLOAT,
        "evidence_ref_indices": [
            VALUE_MAX_APPLIED_EVIDENCE_REFS - 1
            for _ in range(VALUE_MAX_EVIDENCE_REFS)
        ],
        "frozen": True,
        "opposition_count": VALUE_MAX_OPPOSITION_COUNT,
        "origin_admission": _schema_max_enum_value(ValueAdmissionStatus),
        "polarity": -1,
        "strength": _SCHEMA_MAX_UNIT_FLOAT,
    }


def _schema_max_value_revision_record_v7() -> dict[str, object]:
    return {
        "after_digest": _SCHEMA_MAX_DIGEST,
        "after_state_delta": _schema_max_value_revision_delta(),
        "before_digest": _SCHEMA_MAX_DIGEST,
        "event_id": _SCHEMA_MAX_IDENTIFIER,
        "event_sequence": VALUE_MAX_EVENT_SEQUENCE,
        "evidence_ref_indices": [
            VALUE_MAX_APPLIED_EVIDENCE_REFS - 1 for _ in range(VALUE_MAX_REFS)
        ],
        "from_revision": VALUE_MAX_REVISION,
        "operation": _schema_max_enum_value(ValueRevisionOperation),
        "origin_id": _SCHEMA_MAX_DIGEST,
        "previous_record_digest": _SCHEMA_MAX_DIGEST,
        "record_digest": _SCHEMA_MAX_DIGEST,
        "recorded_at": _SCHEMA_MAX_DATETIME,
        "target_revision": VALUE_MAX_REVISION,
        "to_revision": VALUE_MAX_REVISION,
        "value_id": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_value_revision_record_v6() -> dict[str, object]:
    return {
        "after_digest": _SCHEMA_MAX_DIGEST,
        "after_state_projection": _schema_max_value_state(),
        "before_digest": _SCHEMA_MAX_DIGEST,
        "event_id": _SCHEMA_MAX_IDENTIFIER,
        "event_sequence": VALUE_MAX_EVENT_SEQUENCE,
        "evidence_refs": [
            _SCHEMA_MAX_IDENTIFIER for _ in range(VALUE_MAX_REFS)
        ],
        "from_revision": VALUE_MAX_REVISION,
        "operation": _schema_max_enum_value(ValueRevisionOperation),
        "origin_id": _SCHEMA_MAX_DIGEST,
        "previous_record_digest": _SCHEMA_MAX_DIGEST,
        "record_digest": _SCHEMA_MAX_DIGEST,
        "recorded_at": _SCHEMA_MAX_DATETIME,
        "target_revision": VALUE_MAX_REVISION,
        "to_revision": VALUE_MAX_REVISION,
        "value_id": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_value_history_v6() -> dict[str, object]:
    record = _schema_max_value_revision_record_v6()
    return {
        "history_anchor_digest": _SCHEMA_MAX_DIGEST,
        "history_anchor_revision": VALUE_MAX_REVISION,
        "history_anchor_state_digest": _SCHEMA_MAX_DIGEST,
        "records": [record for _ in range(VALUE_MAX_REVISION_RECORDS)],
        "value_id": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_value_history_v7() -> dict[str, object]:
    record = _schema_max_value_revision_record_v7()
    return {
        "history_anchor_digest": _SCHEMA_MAX_DIGEST,
        "history_anchor_revision": VALUE_MAX_REVISION,
        "history_anchor_state_digest": _SCHEMA_MAX_DIGEST,
        "records": [record for _ in range(VALUE_MAX_REVISION_RECORDS)],
        "value_id": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_value_ledger() -> dict[str, object]:
    return {
        "evidence_refs": [
            _SCHEMA_MAX_IDENTIFIER for _ in range(VALUE_MAX_APPLIED_EVIDENCE_REFS)
        ],
        "ledger_digest": _SCHEMA_MAX_DIGEST,
        "value_id": _SCHEMA_MAX_IDENTIFIER,
    }


def _schema_max_value_state_v6() -> dict[str, object]:
    value = _schema_max_value_state()
    history = _schema_max_value_history_v6()
    ledger = _schema_max_value_ledger()
    conflict = {
        "left_value_id": _SCHEMA_MAX_IDENTIFIER,
        "right_value_id": "b" * MAX_IDENTIFIER_CODEPOINTS,
    }
    max_conflicts = VALUE_MAX_AUTHORITATIVE_VALUES * (
        VALUE_MAX_AUTHORITATIVE_VALUES - 1
    ) // 2
    return {
        "conflicts": [conflict for _ in range(max_conflicts)],
        "evidence_ledgers": [
            ledger for _ in range(VALUE_MAX_AUTHORITATIVE_VALUES)
        ],
        "histories": [
            history for _ in range(VALUE_MAX_AUTHORITATIVE_VALUES)
        ],
        "schema_version": 1,
        "values": [value for _ in range(VALUE_MAX_AUTHORITATIVE_VALUES)],
    }


def _schema_max_value_state_v7() -> dict[str, object]:
    value = _schema_max_value_state()
    history = _schema_max_value_history_v7()
    ledger = _schema_max_value_ledger()
    conflict = {
        "left_value_id": _SCHEMA_MAX_IDENTIFIER,
        "right_value_id": "b" * MAX_IDENTIFIER_CODEPOINTS,
    }
    max_conflicts = VALUE_MAX_AUTHORITATIVE_VALUES * (
        VALUE_MAX_AUTHORITATIVE_VALUES - 1
    ) // 2
    return {
        "conflicts": [conflict for _ in range(max_conflicts)],
        "evidence_ledgers": [
            ledger for _ in range(VALUE_MAX_AUTHORITATIVE_VALUES)
        ],
        "histories": [
            history for _ in range(VALUE_MAX_AUTHORITATIVE_VALUES)
        ],
        "schema_version": 2,
        "values": [value for _ in range(VALUE_MAX_AUTHORITATIVE_VALUES)],
    }


def _schema_max_field_bytes() -> dict[str, int]:
    """Return maxima for every current V7 top-level field exactly once."""

    maxima = {
        "appraisal_state": _schema_max_json_bytes(_schema_max_appraisal()),
        "belief_state": BELIEF_SCHEMA_MAX_SERIALIZED_BYTES,
        "context_state": _schema_max_json_bytes(_schema_max_context()),
        "emotion_state": _schema_max_json_bytes(_schema_max_emotion()),
        "last_processed_event_sequence": _schema_max_json_bytes(
            MAX_PERSISTED_EVENT_SEQUENCE
        ),
        "saved_at": _schema_max_json_bytes(_SCHEMA_MAX_DATETIME),
        "schema_version": _schema_max_json_bytes(CURRENT_AGENT_STATE_SCHEMA_VERSION),
        "value_state": _schema_max_json_bytes(_schema_max_value_state_v7()),
        "working_memory": _schema_max_json_bytes(_schema_max_working_memory()),
    }
    if set(maxima) != set(AgentStateSnapshotV7.model_fields):
        raise RuntimeError("AgentState V7 schema maxima are out of sync")
    return maxima


def _schema_max_field_bytes_v6() -> dict[str, int]:
    """Return the retained V6 top-level field maxima without normalization."""

    maxima = {
        "appraisal_state": _schema_max_json_bytes(_schema_max_appraisal()),
        "belief_state": BELIEF_SCHEMA_MAX_SERIALIZED_BYTES,
        "context_state": _schema_max_json_bytes(_schema_max_context()),
        "emotion_state": _schema_max_json_bytes(_schema_max_emotion()),
        "last_processed_event_sequence": _schema_max_json_bytes(
            MAX_PERSISTED_EVENT_SEQUENCE
        ),
        "saved_at": _schema_max_json_bytes(_SCHEMA_MAX_DATETIME),
        "schema_version": _schema_max_json_bytes(6),
        "value_state": _schema_max_json_bytes(_schema_max_value_state_v6()),
        "working_memory": _schema_max_json_bytes(_schema_max_working_memory()),
    }
    if set(maxima) != set(AgentStateSnapshotV6.model_fields):
        raise RuntimeError("AgentState V6 schema maxima are out of sync")
    return maxima


def _schema_envelope_size(field_maxima: Mapping[str, int]) -> int:
    """Count compact JSON braces, keys, colons, and commas exactly."""

    fields = tuple(sorted(field_maxima))
    return 2 + sum(
        len(_canonical_json_bytes(field_name)) + 1 + field_maxima[field_name]
        for field_name in fields
    ) + max(0, len(fields) - 1)


AGENT_STATE_V7_SCHEMA_FIELD_MAX_BYTES: Final[dict[str, int]] = (
    _schema_max_field_bytes()
)
AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES: Final[int] = _schema_envelope_size(
    AGENT_STATE_V7_SCHEMA_FIELD_MAX_BYTES
)
AGENT_STATE_V7_SCHEMA_MAX_SERIALIZED_BYTES: Final[int] = (
    AGENT_STATE_V7_BASE_MAX_SERIALIZED_BYTES
    + AGENT_STATE_FUTURE_STATE_RESERVE_BYTES
)
if AGENT_STATE_V7_SCHEMA_MAX_SERIALIZED_BYTES > AGENT_STATE_MAX_SERIALIZED_BYTES:
    raise RuntimeError("AgentState V7 capacity exceeds the hard serialized byte bound")

# Keep the retained V6 projection available for compatibility accounting.  It
# is not used for new writes, because V7 is the normalized current schema.
AGENT_STATE_V6_SCHEMA_FIELD_MAX_BYTES: Final[dict[str, int]] = (
    _schema_max_field_bytes_v6()
)
AGENT_STATE_V6_SCHEMA_MAX_SERIALIZED_BYTES: Final[int] = (
    _schema_envelope_size(AGENT_STATE_V6_SCHEMA_FIELD_MAX_BYTES)
)


def project_agent_state_schema_max_bytes(
    schema_version: int = CURRENT_AGENT_STATE_SCHEMA_VERSION,
    added_field_maxima: Mapping[str, int] | None = None,
) -> int:
    """Project a future root schema without silently clipping the hard cap.

    ``added_field_maxima`` contains already-serialized JSON VALUE byte
    maxima.  The caller owns both the added field name bound and its value
    bound; this helper adds only the exact quoted-key, colon, and comma
    overhead required by the canonical top-level JSON object.  The returned
    projection includes the future-state reserve and is deliberately not
    clipped to the hard runtime cap; the schema owner must reject an
    infeasible projection.
    """

    if type(schema_version) is not int or schema_version < 0:
        raise ValueError("schema_version must be a non-negative exact integer")
    if added_field_maxima is None:
        additions: Mapping[str, int] = {}
    elif not isinstance(added_field_maxima, Mapping):
        raise TypeError("added_field_maxima must be a mapping")
    else:
        additions = added_field_maxima

    projected = dict(AGENT_STATE_V7_SCHEMA_FIELD_MAX_BYTES)
    projected["schema_version"] = len(_canonical_json_bytes(schema_version))
    for field_name, maximum in additions.items():
        if type(field_name) is not str or not field_name:
            raise ValueError("added field names must be non-empty strings")
        if field_name in projected:
            raise ValueError(f"added field collides with current field: {field_name}")
        if type(maximum) is not int or maximum <= 0:
            raise ValueError(
                "added field maxima must be positive exact integer byte counts"
            )
        projected[field_name] = maximum
    return _schema_envelope_size(projected) + AGENT_STATE_FUTURE_STATE_RESERVE_BYTES


class _LegacyEmotionState(_StateModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    optimal_loss: float = Field(ge=0.0)

    @field_validator("valence", "arousal", "optimal_loss")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("emotion value must be finite")
        return value


class _LegacyAgentStateV0(_StateModel):
    schema_version: Literal[0]
    last_event_sequence: int = Field(ge=0)
    emotion: _LegacyEmotionState

    @field_validator("last_event_sequence", mode="before")
    @classmethod
    def reject_boolean_sequence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("sequence must be an integer")
        return value


class AgentStateError(Exception):
    """Base class for bounded AgentState failures."""


class AgentStateLoadError(AgentStateError):
    """The canonical snapshot exists but cannot be loaded safely."""


class AgentStateConfigurationDrift(AgentStateLoadError):
    """Persisted Value seed/configuration evidence no longer matches settings."""


class UnsupportedAgentStateVersion(AgentStateLoadError):
    """The canonical snapshot uses an unsupported schema version."""


class AgentStateSaveStage(str, Enum):
    CAPTURE = "snapshot_capture"
    TEMP_WRITE = "snapshot_temp_write"
    TEMP_FSYNC = "snapshot_temp_fsync"
    ATOMIC_REPLACE = "snapshot_atomic_replace"
    PARENT_FSYNC = "snapshot_parent_fsync"


class AgentStateSaveError(AgentStateError):
    """A snapshot did not reach confirmed durable success."""

    def __init__(self, stage: AgentStateSaveStage, *, published: bool) -> None:
        self.stage = stage
        self.published = published
        super().__init__(
            "AgentState snapshot save failed "
            f"at {stage.value}; published={str(published).lower()}"
        )


_PRIVATE_STATE_KEYS = frozenset(
    {
        "hiddenthought",
        "privatereasoning",
        "reasoning",
        "chainofthought",
        "thought",
        "content",
        "prompt",
        "rawprompt",
        "systemprompt",
        "userprompt",
        "userinput",
        "assistantprompt",
        "response",
        "retrievedmemory",
        "privatestate",
        "turns",
        "turn",
        "sessionturns",
        "attachments",
        "attachment",
        "eventpayload",
        "requestpayload",
        "debugtrace",
        "debugchattrace",
        "chattranscript",
        "transcript",
    }
)


def _reject_private_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if normalize_private_key(key) in _PRIVATE_STATE_KEYS:
                raise ValueError("snapshot contains a forbidden field")
            _reject_private_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_private_keys(child)


def _origin_snapshot(origin: IdentityOrigin) -> IdentityOriginSnapshot:
    return IdentityOriginSnapshot(
        actor=origin.actor.value,
        input_kind=origin.input_kind.value,
        admission=origin.admission.value,
        source_ref=origin.source_ref,
        event_id=origin.event_id,
        context_id=origin.context_id,
        event_sequence=origin.event_sequence,
        confidence=origin.confidence,
        origin_id=origin.origin_id,
    )


def _value_snapshot(value: ValueState) -> ValueStateSnapshot:
    return ValueStateSnapshot(
        value_id=value.value_id,
        revision=value.revision,
        name=value.name,
        concept=value.concept,
        scope=value.scope.value,
        context_ids=value.context_ids,
        polarity=value.polarity,
        strength=value.strength,
        confidence=value.confidence,
        stability=value.stability,
        protectedness=value.protectedness,
        negotiability=value.negotiability,
        allowed_update_rate=value.allowed_update_rate,
        frozen=value.frozen,
        origin=_origin_snapshot(value.origin),
        evidence_refs=value.evidence_refs,
        seed_contract_digest=value.seed_contract_digest,
        opposition_count=value.opposition_count,
    )


def _revision_record_snapshot(record: ValueRevisionRecord) -> ValueRevisionRecordSnapshot:
    return ValueRevisionRecordSnapshot(
        value_id=record.value_id,
        from_revision=record.from_revision,
        to_revision=record.to_revision,
        before_digest=record.before_digest,
        after_state_projection=_value_snapshot(record.after_state_projection),
        after_digest=record.after_digest,
        operation=record.operation.value,
        origin_id=record.origin_id,
        evidence_refs=record.evidence_refs,
        event_id=record.event_id,
        event_sequence=record.event_sequence,
        recorded_at=record.recorded_at,
        previous_record_digest=record.previous_record_digest,
        target_revision=record.target_revision,
        record_digest=record.record_digest,
    )


def _history_snapshot(history: ValueRevisionHistory) -> ValueRevisionHistorySnapshot:
    return ValueRevisionHistorySnapshot(
        value_id=history.value_id,
        history_anchor_revision=history.history_anchor_revision,
        history_anchor_digest=history.history_anchor_digest,
        history_anchor_state_digest=history.history_anchor_state_digest,
        records=tuple(_revision_record_snapshot(record) for record in history.records),
    )


def _ledger_indices(
    evidence_refs: tuple[str, ...],
    ledger: tuple[str, ...],
) -> tuple[int, ...]:
    positions = {reference: index for index, reference in enumerate(ledger)}
    try:
        return tuple(positions[reference] for reference in evidence_refs)
    except KeyError:
        raise ValueDomainError("Value revision evidence is absent from its ledger") from None


def _revision_delta_snapshot(
    record: ValueRevisionRecord,
    ledger: tuple[str, ...],
) -> ValueRevisionDeltaSnapshot:
    after = record.after_state_projection
    return ValueRevisionDeltaSnapshot(
        polarity=after.polarity,
        strength=after.strength,
        confidence=after.confidence,
        frozen=after.frozen,
        opposition_count=after.opposition_count,
        origin_admission=after.origin.admission.value,
        evidence_ref_indices=_ledger_indices(after.evidence_refs, ledger),
    )


def _revision_record_snapshot_v7(
    record: ValueRevisionRecord,
    ledger: tuple[str, ...],
) -> ValueRevisionRecordSnapshotV7:
    return ValueRevisionRecordSnapshotV7(
        value_id=record.value_id,
        from_revision=record.from_revision,
        to_revision=record.to_revision,
        before_digest=record.before_digest,
        after_state_delta=_revision_delta_snapshot(record, ledger),
        after_digest=record.after_digest,
        operation=record.operation.value,
        origin_id=record.origin_id,
        evidence_ref_indices=_ledger_indices(record.evidence_refs, ledger),
        event_id=record.event_id,
        event_sequence=record.event_sequence,
        recorded_at=record.recorded_at,
        previous_record_digest=record.previous_record_digest,
        target_revision=record.target_revision,
        record_digest=record.record_digest,
    )


def _history_snapshot_v7(
    history: ValueRevisionHistory,
    ledger: tuple[str, ...],
) -> ValueRevisionHistorySnapshotV7:
    # This public owner invariant is the authority for deciding which fields
    # may be factored out of every retained after-state projection.
    validate_value_history_immutable_basis(history)
    return ValueRevisionHistorySnapshotV7(
        value_id=history.value_id,
        history_anchor_revision=history.history_anchor_revision,
        history_anchor_digest=history.history_anchor_digest,
        history_anchor_state_digest=history.history_anchor_state_digest,
        records=tuple(
            _revision_record_snapshot_v7(record, ledger)
            for record in history.records
        ),
    )


def _value_state_snapshot_v7(system: ValueSystem) -> ValueSystemStateSnapshotV7:
    snapshot = system.snapshot()
    ledgers = dict(snapshot.evidence_ledgers)
    digests = dict(snapshot.evidence_ledger_digests)
    return ValueSystemStateSnapshotV7(
        schema_version=2,
        values=tuple(_value_snapshot(value) for value in snapshot.values),
        conflicts=tuple(
            ValueConflictSnapshot(
                left_value_id=conflict.left_value_id,
                right_value_id=conflict.right_value_id,
            )
            for conflict in snapshot.conflicts
        ),
        histories=tuple(
            _history_snapshot_v7(history, ledgers[history.value_id])
            for history in snapshot.histories
        ),
        evidence_ledgers=tuple(
            ValueEvidenceLedgerSnapshot(
                value_id=value_id,
                evidence_refs=ledger,
                ledger_digest=digests[value_id],
            )
            for value_id, ledger in snapshot.evidence_ledgers
        ),
    )


def _value_state_snapshot(system: ValueSystem) -> ValueSystemStateSnapshot:
    snapshot = system.snapshot()
    digests = dict(snapshot.evidence_ledger_digests)
    return ValueSystemStateSnapshot(
        schema_version=1,
        values=tuple(_value_snapshot(value) for value in snapshot.values),
        conflicts=tuple(
            ValueConflictSnapshot(
                left_value_id=conflict.left_value_id,
                right_value_id=conflict.right_value_id,
            )
            for conflict in snapshot.conflicts
        ),
        histories=tuple(_history_snapshot(history) for history in snapshot.histories),
        evidence_ledgers=tuple(
            ValueEvidenceLedgerSnapshot(
                value_id=value_id,
                evidence_refs=ledger,
                ledger_digest=digests[value_id],
            )
            for value_id, ledger in snapshot.evidence_ledgers
        ),
    )


def _belief_proposition_snapshot(
    proposition: BeliefProposition,
) -> BeliefPropositionStateSnapshot:
    return BeliefPropositionStateSnapshot(
        canonical_text=proposition.canonical_text,
        subject=proposition.subject,
        predicate=proposition.predicate,
        object=proposition.object,
        proposition_digest=proposition.proposition_digest,
    )


def _belief_revision_snapshot(
    revision: BeliefRevisionRecord,
) -> BeliefRevisionStateSnapshot:
    if revision.event_id is None or revision.event_sequence is None:
        raise ValueError("authoritative Belief revisions require event evidence")
    return BeliefRevisionStateSnapshot(
        belief_id=revision.belief_id,
        revision=revision.revision,
        operation=revision.operation.value,
        reason=revision.reason.value,
        created_at=revision.created_at,
        previous_revision_digest=revision.previous_revision_digest,
        event_id=revision.event_id,
        event_sequence=revision.event_sequence,
        evidence_refs=revision.evidence_refs,
        record_digest=revision.record_digest,
    )


def _belief_record_snapshot(record: BeliefRecord) -> BeliefRecordStateSnapshot:
    if not record.revision_history or record.revision_history[-1].revision != record.revision:
        raise ValueError("Belief history must include the current revision")
    admission = record.subject_admission
    admission_snapshot = (
        None
        if admission is None
        else BeliefSubjectAdmissionStateSnapshot(
            proposition_digest=admission.proposition_digest,
            evidence_refs=admission.evidence_refs,
            event_id=admission.event_id,
            event_sequence=admission.event_sequence,
            reason=admission.reason.value,
            admission_digest=admission.admission_digest,
        )
    )
    snapshot = BeliefRecordStateSnapshot(
        belief_id=record.belief_id,
        proposition=_belief_proposition_snapshot(record.proposition),
        lifecycle=record.lifecycle.value,
        epistemic_status=record.epistemic_status.value,
        confidence=record.confidence,
        context_scope=record.context_scope,
        valid_from=record.valid_from,
        valid_until=record.valid_until,
        evidence=tuple(
            BeliefEvidenceStateSnapshot(
                evidence_ref=item.evidence_ref,
                evidence_type=item.evidence_type.value,
            )
            for item in record.evidence
        ),
        subject_admission=admission_snapshot,
        supersedes_id=record.supersedes_id,
        superseded_by_id=record.superseded_by_id,
        revision=record.revision,
        revision_history=tuple(
            _belief_revision_snapshot(item) for item in record.revision_history
        ),
        history_anchor_digest=record.history_anchor_digest,
        schema_version=1,
        record_digest=belief_record_digest(record),
    )
    return snapshot


def _belief_state_snapshot(
    snapshot: BeliefSystemSnapshot,
) -> BeliefSystemStateSnapshot:
    if not isinstance(snapshot, BeliefSystemSnapshot):
        raise TypeError("snapshot must be BeliefSystemSnapshot")
    authority = snapshot
    return BeliefSystemStateSnapshot(
        schema_version=1,
        records=tuple(_belief_record_snapshot(record) for record in authority.records),
        authority_digest=authority.authority_digest,
    )


def _domain_belief_proposition(
    snapshot: BeliefPropositionStateSnapshot,
) -> BeliefProposition:
    proposition = BeliefProposition(
        canonical_text=snapshot.canonical_text,
        subject=snapshot.subject,
        predicate=snapshot.predicate,
        object=snapshot.object,
    )
    if proposition.proposition_digest != snapshot.proposition_digest:
        raise ValueError("Belief proposition digest does not match persisted text")
    return proposition


def _domain_belief_revision(
    snapshot: BeliefRevisionStateSnapshot,
) -> BeliefRevisionRecord:
    revision = BeliefRevisionRecord(
        belief_id=snapshot.belief_id,
        revision=snapshot.revision,
        operation=BeliefRevisionOperation(snapshot.operation),
        reason=BeliefRevisionReason(snapshot.reason),
        created_at=snapshot.created_at,
        previous_revision_digest=snapshot.previous_revision_digest,
        event_id=snapshot.event_id,
        event_sequence=snapshot.event_sequence,
        evidence_refs=snapshot.evidence_refs,
    )
    if revision.record_digest != snapshot.record_digest:
        raise ValueError("Belief revision digest does not match persisted record")
    return revision


def _domain_belief_record(snapshot: BeliefRecordStateSnapshot) -> BeliefRecord:
    admission_snapshot = snapshot.subject_admission
    admission = (
        None
        if admission_snapshot is None
        else BeliefSubjectAdmission(
            proposition_digest=admission_snapshot.proposition_digest,
            evidence_refs=admission_snapshot.evidence_refs,
            event_id=admission_snapshot.event_id,
            event_sequence=admission_snapshot.event_sequence,
            reason=BeliefSubjectAdmissionReason(admission_snapshot.reason),
        )
    )
    if admission is not None:
        assert admission_snapshot is not None
        if admission.admission_digest != admission_snapshot.admission_digest:
            raise ValueError("Belief admission digest does not match persisted proof")
    record = BeliefRecord(
        belief_id=snapshot.belief_id,
        proposition=_domain_belief_proposition(snapshot.proposition),
        lifecycle=BeliefLifecycle(snapshot.lifecycle),
        epistemic_status=BeliefEpistemicStatus(snapshot.epistemic_status),
        confidence=snapshot.confidence,
        context_scope=snapshot.context_scope,
        valid_from=snapshot.valid_from,
        valid_until=snapshot.valid_until,
        evidence=tuple(
            BeliefEvidence(
                evidence_ref=item.evidence_ref,
                evidence_type=BeliefEvidenceType(item.evidence_type),
            )
            for item in snapshot.evidence
        ),
        subject_admission=admission,
        supersedes_id=snapshot.supersedes_id,
        superseded_by_id=snapshot.superseded_by_id,
        revision=snapshot.revision,
        revision_history=tuple(
            _domain_belief_revision(item) for item in snapshot.revision_history
        ),
        history_anchor_digest=snapshot.history_anchor_digest,
        schema_version=snapshot.schema_version,
    )
    if belief_record_digest(record) != snapshot.record_digest:
        raise ValueError("Belief record digest does not match persisted state")
    return record


def _domain_belief_system(snapshot: BeliefSystemStateSnapshot) -> BeliefSystem:
    system = BeliefSystem(
        _domain_belief_record(record) for record in snapshot.records
    )
    authority = system.snapshot()
    if (
        snapshot.schema_version != authority.schema_version
        or snapshot.authority_digest != authority.authority_digest
    ):
        raise ValueError("Belief authority digest does not match persisted state")
    return system


def _identity_origin(snapshot: IdentityOriginSnapshot) -> IdentityOrigin:
    origin = IdentityOrigin(
        OriginActor(snapshot.actor),
        OriginInputKind(snapshot.input_kind),
        ValueAdmissionStatus(snapshot.admission),
        source_ref=snapshot.source_ref,
        event_id=snapshot.event_id,
        context_id=snapshot.context_id,
        event_sequence=snapshot.event_sequence,
        confidence=snapshot.confidence,
    )
    if origin.origin_id != snapshot.origin_id:
        raise ValueDomainError("Value origin witness does not match provenance")
    return origin


def _domain_value(snapshot: ValueStateSnapshot) -> ValueState:
    return ValueState(
        value_id=snapshot.value_id,
        revision=snapshot.revision,
        name=snapshot.name,
        concept=snapshot.concept,
        scope=ValueScope(snapshot.scope),
        context_ids=snapshot.context_ids,
        polarity=snapshot.polarity,
        strength=snapshot.strength,
        confidence=snapshot.confidence,
        stability=snapshot.stability,
        protectedness=snapshot.protectedness,
        negotiability=snapshot.negotiability,
        allowed_update_rate=snapshot.allowed_update_rate,
        frozen=snapshot.frozen,
        origin=_identity_origin(snapshot.origin),
        evidence_refs=snapshot.evidence_refs,
        seed_contract_digest=snapshot.seed_contract_digest,
        opposition_count=snapshot.opposition_count,
    )


def _domain_record(snapshot: ValueRevisionRecordSnapshot) -> ValueRevisionRecord:
    record = ValueRevisionRecord(
        value_id=snapshot.value_id,
        from_revision=snapshot.from_revision,
        to_revision=snapshot.to_revision,
        before_digest=snapshot.before_digest,
        after_state_projection=_domain_value(snapshot.after_state_projection),
        after_digest=snapshot.after_digest,
        operation=ValueRevisionOperation(snapshot.operation),
        origin_id=snapshot.origin_id,
        evidence_refs=snapshot.evidence_refs,
        event_id=snapshot.event_id,
        event_sequence=snapshot.event_sequence,
        recorded_at=snapshot.recorded_at,
        previous_record_digest=snapshot.previous_record_digest,
        target_revision=snapshot.target_revision,
    )
    if record.record_digest != snapshot.record_digest:
        raise ValueDomainError("revision record digest does not match persisted record")
    return record


def _domain_history(snapshot: ValueRevisionHistorySnapshot) -> ValueRevisionHistory:
    return ValueRevisionHistory(
        value_id=snapshot.value_id,
        history_anchor_revision=snapshot.history_anchor_revision,
        history_anchor_digest=snapshot.history_anchor_digest,
        history_anchor_state_digest=snapshot.history_anchor_state_digest,
        records=tuple(_domain_record(record) for record in snapshot.records),
    )


def _ledger_references(
    indices: tuple[int, ...], ledger: tuple[str, ...]
) -> tuple[str, ...]:
    if indices != tuple(sorted(set(indices))):
        raise ValueDomainError("Value revision evidence indexes are not canonical")
    if any(index < 0 or index >= len(ledger) for index in indices):
        raise ValueDomainError("Value revision evidence index is outside its ledger")
    return tuple(ledger[index] for index in indices)


def _domain_revision_v7(
    snapshot: ValueRevisionRecordSnapshotV7,
    basis: ValueStateSnapshot,
    ledger: tuple[str, ...],
) -> ValueRevisionRecord:
    if snapshot.value_id != basis.value_id:
        raise ValueDomainError("Value revision basis has a mismatched Value ID")
    delta = snapshot.after_state_delta
    after_origin = basis.origin.model_copy(
        update={"admission": delta.origin_admission}
    )
    after_state = ValueState(
        value_id=basis.value_id,
        revision=snapshot.to_revision,
        name=basis.name,
        concept=basis.concept,
        scope=ValueScope(basis.scope),
        context_ids=basis.context_ids,
        polarity=delta.polarity,
        strength=delta.strength,
        confidence=delta.confidence,
        stability=basis.stability,
        protectedness=basis.protectedness,
        negotiability=basis.negotiability,
        allowed_update_rate=basis.allowed_update_rate,
        frozen=delta.frozen,
        origin=_identity_origin(after_origin),
        evidence_refs=_ledger_references(delta.evidence_ref_indices, ledger),
        seed_contract_digest=basis.seed_contract_digest,
        opposition_count=delta.opposition_count,
    )
    if value_state_digest(after_state) != snapshot.after_digest:
        raise ValueDomainError("Value revision after-state digest does not match")
    record = ValueRevisionRecord(
        value_id=snapshot.value_id,
        from_revision=snapshot.from_revision,
        to_revision=snapshot.to_revision,
        before_digest=snapshot.before_digest,
        after_state_projection=after_state,
        after_digest=snapshot.after_digest,
        operation=ValueRevisionOperation(snapshot.operation),
        origin_id=snapshot.origin_id,
        evidence_refs=_ledger_references(snapshot.evidence_ref_indices, ledger),
        event_id=snapshot.event_id,
        event_sequence=snapshot.event_sequence,
        recorded_at=snapshot.recorded_at,
        previous_record_digest=snapshot.previous_record_digest,
        target_revision=snapshot.target_revision,
    )
    if record.record_digest != snapshot.record_digest:
        raise ValueDomainError("Value revision record digest does not match")
    return record


def _domain_history_v7(
    snapshot: ValueRevisionHistorySnapshotV7,
    basis: ValueStateSnapshot,
    ledger: tuple[str, ...],
) -> ValueRevisionHistory:
    history = ValueRevisionHistory(
        value_id=snapshot.value_id,
        history_anchor_revision=snapshot.history_anchor_revision,
        history_anchor_digest=snapshot.history_anchor_digest,
        history_anchor_state_digest=snapshot.history_anchor_state_digest,
        records=tuple(
            _domain_revision_v7(record, basis, ledger)
            for record in snapshot.records
        ),
    )
    validate_value_history_immutable_basis(history)
    return history


def _domain_value_system_v7(snapshot: ValueSystemStateSnapshotV7) -> ValueSystem:
    values = {
        value.value_id: _domain_value(value) for value in snapshot.values
    }
    ledgers = {
        ledger.value_id: ledger.evidence_refs
        for ledger in snapshot.evidence_ledgers
    }
    ledger_digests = {
        ledger.value_id: ledger.ledger_digest
        for ledger in snapshot.evidence_ledgers
    }
    histories = {
        history.value_id: _domain_history_v7(
            history,
            next(
                value
                for value in snapshot.values
                if value.value_id == history.value_id
            ),
            ledgers[history.value_id],
        )
        for history in snapshot.histories
    }
    conflicts = tuple(
        ValueConflictDefinition(
            left_value_id=conflict.left_value_id,
            right_value_id=conflict.right_value_id,
        )
        for conflict in snapshot.conflicts
    )
    if (
        len(values) != len(snapshot.values)
        or len(histories) != len(snapshot.histories)
        or len(ledgers) != len(snapshot.evidence_ledgers)
        or set(values) != set(histories)
        or set(values) != set(ledgers)
    ):
        raise ValueDomainError("normalized Value snapshot keys are inconsistent")
    return ValueSystem.restore(
        values=values,
        conflicts=conflicts,
        histories=histories,
        evidence_ledgers=ledgers,
        evidence_ledger_digests=ledger_digests,
    )


def _domain_value_system(
    snapshot: ValueSystemStateSnapshot | ValueSystemStateSnapshotV7,
) -> ValueSystem:
    if isinstance(snapshot, ValueSystemStateSnapshotV7):
        try:
            return _domain_value_system_v7(snapshot)
        except Exception:
            raise AgentStateLoadError("ValueSystem snapshot is invalid") from None
    try:
        values = {_value.value_id: _domain_value(_value) for _value in snapshot.values}
        histories = {
            _history.value_id: _domain_history(_history)
            for _history in snapshot.histories
        }
        ledgers = {
            ledger.value_id: ledger.evidence_refs
            for ledger in snapshot.evidence_ledgers
        }
        ledger_digests = {
            ledger.value_id: ledger.ledger_digest
            for ledger in snapshot.evidence_ledgers
        }
        if len(values) != len(snapshot.values):
            raise ValueDomainError("Value snapshot contains duplicate Value IDs")
        if len(histories) != len(snapshot.histories):
            raise ValueDomainError("Value snapshot contains duplicate history IDs")
        if len(ledgers) != len(snapshot.evidence_ledgers):
            raise ValueDomainError("Value snapshot contains duplicate ledger IDs")
        conflicts = tuple(
            ValueConflictDefinition(
                left_value_id=conflict.left_value_id,
                right_value_id=conflict.right_value_id,
            )
            for conflict in snapshot.conflicts
        )
        return ValueSystem.restore(
            values=values,
            conflicts=conflicts,
            histories=histories,
            evidence_ledgers=ledgers,
            evidence_ledger_digests=ledger_digests,
        )
    except Exception:
        pass
    raise AgentStateLoadError("ValueSystem snapshot is invalid")


def _upgrade_value_state_to_v7(
    snapshot: ValueSystemStateSnapshot | ValueSystemStateSnapshotV7,
) -> ValueSystemStateSnapshotV7:
    if isinstance(snapshot, ValueSystemStateSnapshotV7):
        _domain_value_system_v7(snapshot)
        return snapshot
    return _value_state_snapshot_v7(_domain_value_system(snapshot))


def _upgrade_snapshot_to_v7(
    snapshot: CompatibleAgentStateSnapshot,
) -> AgentStateSnapshotV7:
    if isinstance(snapshot, AgentStateSnapshotV7):
        _domain_value_system(snapshot.value_state)
        _domain_belief_system(snapshot.belief_state)
        return snapshot
    if isinstance(snapshot, AgentStateSnapshotV6):
        return AgentStateSnapshotV7(
            saved_at=snapshot.saved_at,
            last_processed_event_sequence=snapshot.last_processed_event_sequence,
            emotion_state=snapshot.emotion_state,
            working_memory=snapshot.working_memory,
            context_state=snapshot.context_state,
            appraisal_state=snapshot.appraisal_state,
            value_state=_upgrade_value_state_to_v7(snapshot.value_state),
            belief_state=snapshot.belief_state,
        )
    if isinstance(snapshot, AgentStateSnapshotV5):
        return AgentStateSnapshotV7(
            saved_at=snapshot.saved_at,
            last_processed_event_sequence=snapshot.last_processed_event_sequence,
            emotion_state=snapshot.emotion_state,
            working_memory=snapshot.working_memory,
            context_state=snapshot.context_state,
            appraisal_state=snapshot.appraisal_state,
            value_state=_upgrade_value_state_to_v7(snapshot.value_state),
            belief_state=_belief_state_snapshot(BeliefSystem().snapshot()),
        )
    raise AgentStateLoadError("Only v5 and v6 snapshots can be upgraded to v7")


def default_agent_state_snapshot(
    baseline_surprisal: float,
    *,
    saved_at: datetime | None = None,
    value_system: ValueSystem | None = None,
) -> AgentStateSnapshotV7:
    """Return the bootstrap state used only when the canonical file is absent."""

    system = value_system if value_system is not None else ValueSystem()
    return AgentStateSnapshotV7(
        saved_at=saved_at or datetime.now(timezone.utc),
        last_processed_event_sequence=0,
        emotion_state=EmotionStateSnapshot(
            valence=0.0,
            arousal=0.0,
            optimal_loss=baseline_surprisal,
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
        context_state=ContextStateSnapshot(
            revision=0,
            current_context_id=None,
            frames=(),
            interlocutor_bindings=(),
        ),
        appraisal_state=AppraisalStateSnapshot(
            calibration_entries=(), last_emotion_update_at=None
        ),
        value_state=_value_state_snapshot_v7(system),
        belief_state=_belief_state_snapshot(BeliefSystem().snapshot()),
    )


def _value_system_authority(main_loop: SuzkaMainLoop) -> ValueSystem | None:
    """Read the internal Value authority without exposing it as a public port."""

    getter = getattr(main_loop, "_value_system_for_state", None)
    if callable(getter):
        value_system = getter()
    else:
        value_system = getattr(main_loop, "value_system", None)
    return value_system if isinstance(value_system, ValueSystem) else None


def _replace_value_system_authority(
    main_loop: SuzkaMainLoop, value_system: ValueSystem
) -> None:
    """Replace Value authority through the MainLoop state boundary when present."""

    setter = getattr(main_loop, "_replace_value_system_for_state", None)
    if callable(setter):
        setter(value_system)
    else:
        setattr(main_loop, "value_system", value_system)


def _belief_state_port(main_loop: SuzkaMainLoop) -> BeliefStatePort | None:
    """Return the explicit runtime-owned Belief state port, if implemented."""

    return main_loop if isinstance(main_loop, BeliefStatePort) else None


class AgentStateStore:
    """Persist runtime state without taking over domain authority.

    AgentStateStore owns snapshot schema, migration, publication, and rollback
    continuity.  BeliefSystem remains the owner of Belief mutation and
    lifecycle authority behind ``BeliefStatePort``.
    """

    def __init__(
        self,
        path: str | Path,
        baseline_surprisal: float,
        *,
        value_seeds: tuple[ValueSeedDeclaration, ...] = (),
        value_conflicts: tuple[ValueConflictDefinition, ...] = (),
        clock: Callable[[], datetime] | None = None,
        save_stage_hook: Callable[[AgentStateSaveStage], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._baseline_surprisal = baseline_surprisal
        self._value_seeds: tuple[ValueSeedDeclaration, ...] = ()
        self._value_conflicts: tuple[ValueConflictDefinition, ...] = ()
        self._configured_value_system = ValueSystem()
        self.configure_value_contract(value_seeds, value_conflicts)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._save_stage_hook = save_stage_hook

    @property
    def configured_value_system(self) -> ValueSystem:
        """Return a fresh configured baseline for legacy/fresh startup."""

        return ValueSystem.restore_snapshot(self._configured_value_system.snapshot())

    def configure_value_contract(
        self,
        value_seeds: tuple[ValueSeedDeclaration, ...],
        value_conflicts: tuple[ValueConflictDefinition, ...],
    ) -> None:
        """Bind immutable configuration evidence before loading durable state."""

        seeds = tuple(value_seeds)
        conflicts = tuple(value_conflicts)
        if any(not isinstance(seed, ValueSeedDeclaration) for seed in seeds):
            raise TypeError("value_seeds must contain ValueSeedDeclaration values")
        if any(not isinstance(conflict, ValueConflictDefinition) for conflict in conflicts):
            raise TypeError(
                "value_conflicts must contain ValueConflictDefinition values"
            )
        conflicts = tuple(
            sorted(
                conflicts,
                key=lambda conflict: (
                    conflict.left_value_id,
                    conflict.right_value_id,
                ),
            )
        )
        configured = ValueSystem.from_seed_declarations(seeds, conflicts)
        self._value_seeds = seeds
        self._value_conflicts = conflicts
        self._configured_value_system = configured

    def snapshot_exists(self) -> bool:
        """Inspect canonical snapshot presence without following its final path."""

        try:
            self.path.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            raise AgentStateLoadError(
                "AgentState snapshot cannot be inspected"
            ) from None
        return True

    def load(self) -> CompatibleAgentStateSnapshot:
        inspection_failure: AgentStateLoadError | None = None
        try:
            path_status = self.path.lstat()
        except FileNotFoundError:
            try:
                return default_agent_state_snapshot(
                    self._baseline_surprisal,
                    saved_at=self._now(),
                    value_system=self.configured_value_system,
                )
            except Exception:
                inspection_failure = AgentStateLoadError("AgentState bootstrap failed")
        except OSError:
            inspection_failure = AgentStateLoadError(
                "AgentState snapshot cannot be inspected"
            )
        if inspection_failure is not None:
            raise inspection_failure

        if not stat.S_ISREG(path_status.st_mode):
            raise AgentStateLoadError("AgentState snapshot is not a regular file")

        read_failure: AgentStateLoadError | None = None
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                file_stat = os.fstat(descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise OSError("snapshot is not a regular file")
                if file_stat.st_size > AGENT_STATE_MAX_SERIALIZED_BYTES:
                    raise AgentStateLoadError(
                        "AgentState snapshot exceeds its serialized byte bound"
                    )
                with os.fdopen(descriptor, "rb") as snapshot_file:
                    descriptor = -1
                    raw_bytes = snapshot_file.read(AGENT_STATE_MAX_SERIALIZED_BYTES + 1)
                    if len(raw_bytes) > AGENT_STATE_MAX_SERIALIZED_BYTES:
                        raise AgentStateLoadError(
                            "AgentState snapshot exceeds its serialized byte bound"
                        )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            raw = json.loads(
                raw_bytes.decode("utf-8"),
                parse_constant=self._reject_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            read_failure = AgentStateLoadError("AgentState snapshot is malformed")
        if read_failure is not None:
            raise read_failure

        if not isinstance(raw, dict):
            raise AgentStateLoadError("AgentState snapshot root is invalid")
        privacy_failure: AgentStateLoadError | None = None
        try:
            _reject_private_keys(raw)
        except ValueError:
            privacy_failure = AgentStateLoadError(
                "AgentState snapshot violates privacy"
            )
        if privacy_failure is not None:
            raise privacy_failure

        version = raw.get("schema_version")
        if version == 1:
            schema_failure: AgentStateLoadError | None = None
            try:
                return AgentStateSnapshotV1.model_validate(raw)
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == 2:
            schema_failure = None
            try:
                return AgentStateSnapshotV2.model_validate(raw)
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == 3:
            try:
                return AgentStateSnapshotV3.model_validate(raw)
            except ValidationError:
                raise AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                ) from None
        if version == 4:
            try:
                return AgentStateSnapshotV4.model_validate(raw)
            except ValidationError:
                raise AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                ) from None
        if version == 5:
            schema_failure = None
            try:
                loaded_v5 = AgentStateSnapshotV5.model_validate(raw)
                self._validate_value_configuration(loaded_v5)
                _domain_value_system(loaded_v5.value_state)
                return loaded_v5
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            except AgentStateLoadError:
                raise
            except Exception:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == 6:
            schema_failure = None
            try:
                loaded_v6 = AgentStateSnapshotV6.model_validate(raw)
                self._validate_value_configuration(loaded_v6)
                _domain_value_system(loaded_v6.value_state)
                _domain_belief_system(loaded_v6.belief_state)
                return loaded_v6
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            except AgentStateLoadError:
                raise
            except Exception:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == CURRENT_AGENT_STATE_SCHEMA_VERSION:
            schema_failure = None
            try:
                loaded_v7 = AgentStateSnapshotV7.model_validate(raw)
                self._validate_value_configuration(loaded_v7)
                _domain_value_system(loaded_v7.value_state)
                _domain_belief_system(loaded_v7.belief_state)
                return loaded_v7
            except ValidationError:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            except AgentStateLoadError:
                raise
            except Exception:
                schema_failure = AgentStateLoadError(
                    "AgentState snapshot schema is invalid"
                )
            raise schema_failure
        if version == 0:
            return self._migrate_v0(raw)
        if isinstance(version, int) and not isinstance(version, bool):
            raise UnsupportedAgentStateVersion(
                "AgentState schema version is unsupported"
            )
        raise AgentStateLoadError("AgentState schema version is invalid")

    def _validate_value_configuration(
        self,
        snapshot: AgentStateSnapshotV5
        | AgentStateSnapshotV6
        | AgentStateSnapshotV7,
    ) -> None:
        """Check configuration as compatibility evidence, never as overwrite authority."""

        value_system = _domain_value_system(snapshot.value_state)
        self._validate_value_system_configuration(value_system)

    def _validate_value_system_configuration(self, value_system: ValueSystem) -> None:
        """Check configured seed lineage without adopting or mutating Values."""

        persisted = value_system.value_map
        for seed in self._value_seeds:
            value = persisted.get(seed.value_id)
            if value is None:
                # A newly declared seed is not adopted into an existing snapshot.
                continue
            if value.origin.admission is not ValueAdmissionStatus.SYSTEM_AUTHORIZED:
                raise AgentStateConfigurationDrift(
                    "Value configuration collides with persisted non-system state"
                )
            if value.seed_contract_digest != recompute_seed_contract_digest(seed):
                raise AgentStateConfigurationDrift(
                    "Persisted system Value seed declaration has drifted"
                )

    def save(self, snapshot: CompatibleAgentStateSnapshot) -> None:
        stage = AgentStateSaveStage.TEMP_WRITE
        published = False
        temporary_path: Path | None = None
        descriptor: int | None = None
        save_failure: AgentStateSaveError | None = None
        try:
            payload = self.canonical_bytes(snapshot)
            parent = self.path.parent
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=parent
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as temporary_file:
                descriptor = None
                self._run_stage(stage)
                if temporary_file.write(payload) != len(payload):
                    raise OSError("incomplete snapshot write")
                temporary_file.flush()
                stage = AgentStateSaveStage.TEMP_FSYNC
                self._run_stage(stage)
                os.fsync(temporary_file.fileno())

            stage = AgentStateSaveStage.ATOMIC_REPLACE
            self._run_stage(stage)
            os.replace(temporary_path, self.path)
            published = True

            stage = AgentStateSaveStage.PARENT_FSYNC
            self._run_stage(stage)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except Exception:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path is not None and not published:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
            save_failure = AgentStateSaveError(stage, published=published)
        if save_failure is not None:
            raise save_failure

    def canonical_bytes(self, snapshot: CompatibleAgentStateSnapshot) -> bytes:
        """Return the one canonical representation used for save and hashing."""

        canonical_failure: AgentStateSaveError | None = None
        try:
            raw: object = snapshot.model_dump(mode="python")
            _reject_private_keys(raw)
            validated = validate_compatible_agent_state_snapshot(raw)
            if isinstance(
                validated,
                (AgentStateSnapshotV5, AgentStateSnapshotV6, AgentStateSnapshotV7),
            ):
                self._validate_value_configuration(validated)
                _domain_value_system(validated.value_state)
            if isinstance(validated, (AgentStateSnapshotV6, AgentStateSnapshotV7)):
                _domain_belief_system(validated.belief_state)
            payload = self._canonical_bytes(validated)
            if len(payload) > AGENT_STATE_MAX_SERIALIZED_BYTES:
                raise ValueError("AgentState snapshot exceeds its serialized byte bound")
            return payload
        except (AgentStateConfigurationDrift, AgentStateLoadError):
            raise
        except Exception:
            canonical_failure = AgentStateSaveError(
                AgentStateSaveStage.CAPTURE, published=False
            )
        raise canonical_failure

    def snapshot_hash(self, snapshot: CompatibleAgentStateSnapshot) -> str:
        """Hash the exact canonical bytes published by this store."""

        return hashlib.sha256(self.canonical_bytes(snapshot)).hexdigest()

    def ensure_published(self, snapshot: CompatibleAgentStateSnapshot) -> None:
        """Publish bootstrap/migrated state while avoiding an identical rewrite."""

        try:
            preserve_legacy = (
                isinstance(
                    snapshot,
                    (
                        AgentStateSnapshotV1,
                        AgentStateSnapshotV2,
                        AgentStateSnapshotV3,
                        AgentStateSnapshotV4,
                        AgentStateSnapshotV5,
                        AgentStateSnapshotV6,
                    ),
                )
                and self.snapshot_exists()
            )
        except AgentStateLoadError:
            preserve_legacy = False
        if preserve_legacy:
            try:
                published_snapshot = self.load()
            except AgentStateLoadError:
                pass
            else:
                if published_snapshot == snapshot:
                    return
        payload = self.canonical_bytes(snapshot)
        inspection_failure: AgentStateSaveError | None = None
        descriptor: int | None = None
        try:
            status = self.path.lstat()
        except FileNotFoundError:
            self.save(snapshot)
            return
        except OSError:
            inspection_failure = AgentStateSaveError(
                AgentStateSaveStage.TEMP_WRITE, published=False
            )
        else:
            try:
                if not stat.S_ISREG(status.st_mode):
                    raise OSError("snapshot is not a regular file")
                descriptor = os.open(
                    self.path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                file_stat = os.fstat(descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise OSError("snapshot is not a regular file")
                if file_stat.st_size > AGENT_STATE_MAX_SERIALIZED_BYTES:
                    raise OSError("snapshot exceeds its serialized byte bound")
                with os.fdopen(descriptor, "rb") as snapshot_file:
                    descriptor = None
                    published = snapshot_file.read(AGENT_STATE_MAX_SERIALIZED_BYTES + 1)
                    if len(published) > AGENT_STATE_MAX_SERIALIZED_BYTES:
                        raise OSError("snapshot exceeds its serialized byte bound")
            except OSError:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                inspection_failure = AgentStateSaveError(
                    AgentStateSaveStage.TEMP_WRITE, published=False
                )
            else:
                if published == payload:
                    return
        if inspection_failure is not None:
            raise inspection_failure
        self.save(snapshot)

    def capture(
        self, main_loop: SuzkaMainLoop, sequence: int
    ) -> AgentStateSnapshotV5 | AgentStateSnapshotV7:
        capture_failure: AgentStateSaveError | None = None
        try:
            emotion_engine = getattr(main_loop, "emotion_engine", None)
            if not isinstance(emotion_engine, EmotionEngineAllostasis):
                raise ValueError("Emotion authority is unavailable")
            working_memory = getattr(main_loop, "working_memory", None)
            if not isinstance(working_memory, WorkingMemory):
                raise ValueError("Working Memory authority is unavailable")
            context_registry = getattr(main_loop, "context_registry", None)
            if not isinstance(context_registry, ContextRegistry):
                raise ValueError("Context authority is unavailable")
            emotion = emotion_engine.state
            context_state = context_registry.state
            validate_context_registry_state(context_state)
            calibration = getattr(main_loop, "loss_calibration")
            if not isinstance(calibration, LossCalibration):
                raise ValueError("Calibration authority is unavailable")
            exported_calibration = calibration.export()
            if not isinstance(exported_calibration, tuple):
                raise ValueError("Calibration authority is malformed")
            calibration_entries = tuple(
                CalibrationEntrySnapshot(
                    model_key=entry.model_key,
                    count=entry.count,
                    mean=entry.mean,
                    m2=entry.m2,
                )
                for entry in exported_calibration
            )
            temporal = getattr(emotion_engine, "temporal_state")
            if not isinstance(temporal, EmotionTemporalState):
                raise ValueError("Emotion temporal authority is malformed")
            value_system = _value_system_authority(main_loop)
            if not isinstance(value_system, ValueSystem):
                raise ValueError("Value authority is unavailable")
            value_system.validate()
            self._validate_value_system_configuration(value_system)
            belief_port = _belief_state_port(main_loop)
            belief_snapshot = (
                belief_port.export_belief_state() if belief_port is not None else None
            )
            if belief_snapshot is not None and not isinstance(
                belief_snapshot, BeliefSystemSnapshot
            ):
                raise ValueError("Belief state port returned an invalid snapshot")
            common: dict[str, Any] = dict(
                saved_at=self._now(),
                last_processed_event_sequence=sequence,
                emotion_state=EmotionStateSnapshot(
                    valence=emotion.valence,
                    arousal=emotion.arousal,
                    optimal_loss=emotion.optimal_loss,
                ),
                working_memory=WorkingMemorySnapshot(
                    revision=working_memory.revision,
                    items=tuple(
                        WorkingMemoryItemSnapshot(
                            item_id=item.item_id,
                            source_kind=item.source_kind.value,
                            source_id=item.source_id,
                            activation=item.activation,
                            salience=item.salience,
                            retention_reason=item.retention_reason.value,
                            created_revision=item.created_revision,
                            last_activated_revision=item.last_activated_revision,
                        )
                        for item in working_memory.items
                    ),
                ),
                context_state=ContextStateSnapshot(
                    revision=context_state.revision,
                    current_context_id=context_state.current_context_id,
                    frames=tuple(
                        ContextFrameSnapshot(
                            context_id=frame.context_id,
                            context_type=cast(
                                Literal["conversation"], frame.context_type.value
                            ),
                            source_channel=frame.source_channel,
                            source_session_id=frame.source_session_id,
                            participant_refs=frame.participant_refs,
                            parent_context_id=frame.parent_context_id,
                            related_context_ids=frame.related_context_ids,
                            status=cast(
                                Literal["active", "suspended", "closed"],
                                frame.status.value,
                            ),
                            created_revision=frame.created_revision,
                            last_modified_revision=frame.last_modified_revision,
                            started_at=frame.started_at,
                            last_active_at=frame.last_active_at,
                        )
                        for frame in context_state.frames
                    ),
                    interlocutor_bindings=tuple(
                        InterlocutorBindingSnapshot(
                            reference_key=binding.reference_key,
                            identity_key=binding.identity_key,
                            confidence=binding.confidence,
                            evidence_references=binding.evidence_references,
                            created_revision=binding.created_revision,
                            last_modified_revision=binding.last_modified_revision,
                        )
                        for binding in context_state.interlocutor_bindings
                    ),
                ),
                appraisal_state=AppraisalStateSnapshot(
                    calibration_entries=calibration_entries,
                    last_emotion_update_at=temporal.last_update_at,
                ),
            )
            if belief_snapshot is None:
                # A runtime without the explicit Belief port cannot authoritatively
                # claim an empty Belief state.  Preserve the pre-Belief v5 shape
                # until a runtime with that port performs the next commit.
                return AgentStateSnapshotV5(
                    **common,
                    value_state=_value_state_snapshot(value_system),
                )
            value_state = _value_state_snapshot_v7(value_system)
            return AgentStateSnapshotV7(
                **common,
                value_state=value_state,
                belief_state=_belief_state_snapshot(belief_snapshot),
            )
        except Exception:
            capture_failure = AgentStateSaveError(
                AgentStateSaveStage.CAPTURE, published=False
            )
        raise capture_failure

    def restore_into(
        self, main_loop: SuzkaMainLoop, snapshot: CompatibleAgentStateSnapshot
    ) -> None:
        restore_failure: AgentStateLoadError | None = None
        context_registry: ContextRegistry | None = None
        calibration: LossCalibration | None = None
        previous_emotion: EmotionState | None = None
        previous_working_memory_revision: int | None = None
        previous_working_memory_items: tuple[WorkingMemoryItem, ...] | None = None
        previous_context: ContextRegistryState | None = None
        previous_calibration: tuple[CalibrationEntry, ...] | None = None
        previous_temporal: EmotionTemporalState | None = None
        previous_value_system: ValueSystem | None = None
        previous_belief_snapshot: BeliefSystemSnapshot | None = None
        value_system_authority: ValueSystem | None = None
        belief_port: BeliefStatePort | None = None
        value_system_was_present = False
        belief_port_was_present = False
        emotion_engine: EmotionEngineAllostasis | None = None
        working_memory_authority: WorkingMemory | None = None
        state_mutation_started = False
        try:
            validated = validate_compatible_agent_state_snapshot(
                snapshot.model_dump(mode="python")
            )
            if isinstance(
                validated,
                (AgentStateSnapshotV5, AgentStateSnapshotV6, AgentStateSnapshotV7),
            ):
                self._validate_value_configuration(validated)
                restored_value_system = _domain_value_system(validated.value_state)
            else:
                restored_value_system = self.configured_value_system
            restored_belief_system = (
                _domain_belief_system(validated.belief_state)
                if isinstance(validated, (AgentStateSnapshotV6, AgentStateSnapshotV7))
                else BeliefSystem()
            )
            emotion_engine = getattr(main_loop, "emotion_engine", None)
            if not isinstance(emotion_engine, EmotionEngineAllostasis):
                raise AgentStateLoadError("AgentState restore requires EmotionEngine")
            working_memory_authority = getattr(main_loop, "working_memory", None)
            if not isinstance(working_memory_authority, WorkingMemory):
                raise AgentStateLoadError("AgentState restore requires WorkingMemory")
            emotion = validated.emotion_state
            working_memory = (
                validated.working_memory
                if isinstance(
                    validated,
                    (
                        AgentStateSnapshotV2,
                        AgentStateSnapshotV3,
                        AgentStateSnapshotV4,
                        AgentStateSnapshotV5,
                        AgentStateSnapshotV6,
                        AgentStateSnapshotV7,
                    ),
                )
                else WorkingMemorySnapshot(revision=0, items=())
            )
            context_registry = getattr(main_loop, "context_registry", None)
            if context_registry is not None and not isinstance(
                context_registry, ContextRegistry
            ):
                raise AgentStateLoadError("AgentState restore requires ContextRegistry")
            context_state = (
                validated.context_state.to_registry_state()
                if isinstance(
                    validated,
                    (
                        AgentStateSnapshotV3,
                        AgentStateSnapshotV4,
                        AgentStateSnapshotV5,
                        AgentStateSnapshotV6,
                        AgentStateSnapshotV7,
                    ),
                )
                else ContextRegistryState(0, None, (), ())
            )
            validate_context_registry_state(context_state)
            calibration = getattr(main_loop, "loss_calibration", None)
            if not isinstance(calibration, LossCalibration):
                raise AgentStateLoadError("AgentState restore requires LossCalibration")
            appraisal = (
                validated.appraisal_state
                if isinstance(
                    validated,
                    (
                        AgentStateSnapshotV4,
                        AgentStateSnapshotV5,
                        AgentStateSnapshotV6,
                        AgentStateSnapshotV7,
                    ),
                )
                else AppraisalStateSnapshot(
                    calibration_entries=(), last_emotion_update_at=None
                )
            )
            restored_calibration = tuple(
                entry.to_entry() for entry in appraisal.calibration_entries
            )
            restored_temporal = EmotionTemporalState(appraisal.last_emotion_update_at)
            current_temporal = getattr(emotion_engine, "temporal_state", None)
            if not isinstance(current_temporal, EmotionTemporalState):
                raise AgentStateLoadError(
                    "AgentState restore requires EmotionTemporalState"
                )
            restored_items = tuple(
                WorkingMemoryItem(
                    item_id=item.item_id,
                    source_kind=WorkingMemorySourceKind(item.source_kind),
                    source_id=item.source_id,
                    activation=item.activation,
                    salience=item.salience,
                    retention_reason=WorkingMemoryRetentionReason(
                        item.retention_reason
                    ),
                    created_revision=item.created_revision,
                    last_activated_revision=item.last_activated_revision,
                )
                for item in working_memory.items
            )
            if (
                isinstance(
                    validated,
                    (
                        AgentStateSnapshotV3,
                        AgentStateSnapshotV4,
                        AgentStateSnapshotV5,
                        AgentStateSnapshotV6,
                        AgentStateSnapshotV7,
                    ),
                )
                and context_registry is None
            ):
                raise AgentStateLoadError("AgentState restore requires ContextRegistry")

            current_value_system = _value_system_authority(main_loop)
            if isinstance(
                validated,
                (AgentStateSnapshotV5, AgentStateSnapshotV6, AgentStateSnapshotV7),
            ):
                if not isinstance(current_value_system, ValueSystem):
                    raise AgentStateLoadError(
                        "AgentState restore requires ValueSystem authority"
                    )
                value_system_authority = current_value_system
                previous_value_system = current_value_system
                value_system_was_present = True
            elif isinstance(current_value_system, ValueSystem):
                value_system_authority = current_value_system
                previous_value_system = current_value_system
                value_system_was_present = True

            belief_port = _belief_state_port(main_loop)
            if isinstance(validated, AgentStateSnapshotV6) or (
                isinstance(validated, AgentStateSnapshotV7)
                and bool(validated.belief_state.records)
            ):
                if belief_port is None:
                    raise AgentStateLoadError(
                        "AgentState restore requires BeliefSystem authority via the explicit state port"
                    )
            if belief_port is not None:
                previous_belief_snapshot = belief_port.export_belief_state()
                if not isinstance(previous_belief_snapshot, BeliefSystemSnapshot):
                    raise AgentStateLoadError("Belief state port returned an invalid snapshot")
                belief_port_was_present = True

            previous_emotion = emotion_engine.state
            previous_working_memory_revision = working_memory_authority.revision
            previous_working_memory_items = working_memory_authority.items
            previous_context = (
                context_registry.state if context_registry is not None else None
            )
            previous_calibration = calibration.export()
            previous_temporal = current_temporal
            state_mutation_started = True
            working_memory_authority.restore_exact(
                working_memory.revision,
                restored_items,
            )
            if context_registry is not None:
                context_registry.restore_exact(context_state)
            calibration.restore_exact(restored_calibration)
            if value_system_authority is not None:
                _replace_value_system_authority(main_loop, restored_value_system)
            if belief_port is not None:
                belief_port.restore_belief_state(restored_belief_system.snapshot())
            emotion_engine.state = EmotionState(
                valence=emotion.valence,
                arousal=emotion.arousal,
                optimal_loss=emotion.optimal_loss,
            )
            emotion_engine.temporal_state = restored_temporal
        except AgentStateConfigurationDrift:
            raise
        except Exception as error:
            if isinstance(error, AgentStateLoadError) and not state_mutation_started:
                raise
            if (
                previous_working_memory_revision is not None
                and previous_working_memory_items is not None
                and working_memory_authority is not None
            ):
                try:
                    working_memory_authority.restore_exact(
                        previous_working_memory_revision,
                        previous_working_memory_items,
                    )
                except Exception:
                    pass
            if previous_context is not None and context_registry is not None:
                try:
                    context_registry.restore_exact(previous_context)
                except Exception:
                    pass
            if previous_calibration is not None and calibration is not None:
                try:
                    calibration.restore_exact(previous_calibration)
                except Exception:
                    pass
            if previous_emotion is not None and emotion_engine is not None:
                try:
                    emotion_engine.state = previous_emotion
                except Exception:
                    pass
            if previous_temporal is not None and emotion_engine is not None:
                try:
                    emotion_engine.temporal_state = previous_temporal
                except Exception:
                    pass
            if value_system_was_present and previous_value_system is not None:
                try:
                    _replace_value_system_authority(main_loop, previous_value_system)
                except Exception:
                    pass
            if belief_port_was_present and previous_belief_snapshot is not None:
                try:
                    assert belief_port is not None
                    belief_port.restore_belief_state(previous_belief_snapshot)
                except Exception:
                    pass
            restore_failure = AgentStateLoadError("AgentState restore failed")
        if restore_failure is not None:
            raise restore_failure

    def _migrate_v0(self, raw: dict[str, Any]) -> AgentStateSnapshotV7:
        migration_failure: AgentStateLoadError | None = None
        try:
            legacy = _LegacyAgentStateV0.model_validate(raw)
            return default_agent_state_snapshot(
                legacy.emotion.optimal_loss,
                saved_at=self._now(),
                value_system=self.configured_value_system,
            ).model_copy(
                update={
                    "last_processed_event_sequence": legacy.last_event_sequence,
                    "emotion_state": EmotionStateSnapshot(
                        valence=legacy.emotion.valence,
                        arousal=legacy.emotion.arousal,
                        optimal_loss=legacy.emotion.optimal_loss,
                    ),
                }
            )
        except Exception:
            migration_failure = AgentStateLoadError("AgentState v0 migration failed")
        raise migration_failure

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("AgentState clock must return timezone-aware UTC")
        return value.astimezone(timezone.utc)

    def _run_stage(self, stage: AgentStateSaveStage) -> None:
        if self._save_stage_hook is not None:
            self._save_stage_hook(stage)

    @staticmethod
    def _canonical_bytes(snapshot: CompatibleAgentStateSnapshot) -> bytes:
        if isinstance(
            snapshot,
            (AgentStateSnapshotV5, AgentStateSnapshotV6, AgentStateSnapshotV7),
        ):
            _domain_value_system(snapshot.value_state)
        if isinstance(snapshot, (AgentStateSnapshotV6, AgentStateSnapshotV7)):
            _domain_belief_system(snapshot.belief_state)
        payload = json.dumps(
            snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > AGENT_STATE_MAX_SERIALIZED_BYTES:
            raise ValueError("AgentState snapshot exceeds its serialized byte bound")
        return payload

    @staticmethod
    def _reject_json_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")
