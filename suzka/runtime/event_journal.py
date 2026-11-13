"""Durable, integrity-chained lifecycle evidence for agent events."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import fcntl
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType


CURRENT_EVENT_JOURNAL_SCHEMA_VERSION: Literal[4] = 4
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN_V1 = b"PROJECT-SUZKA:event-journal:v1\0"
_HASH_DOMAIN_V2 = b"PROJECT-SUZKA:event-journal:v2\0"
LOGGER = logging.getLogger(__name__)
_HASH_DOMAIN_V3 = b"PROJECT-SUZKA:event-journal:v3\0"
_HASH_DOMAIN_V4 = b"PROJECT-SUZKA:event-journal:v4\0"
_ARCHIVE_ROOT_DOMAIN = b"SUZKA:R07:JOURNAL-ARCHIVE:V1\0"
_ARCHIVE_NAME_PATTERN = re.compile(r"(\d+)-(\d+)-([0-9a-f-]{36})\.jsonl\.gz")
# Fields introduced by schema 4. They are absent from (and excluded when
# hashing or serialising) every record of an older schema, so v1-v3 record
# hashes and bytes are unchanged.
_V4_FIELDS = (
    "migration_previous_v3_hash",
    "v4_migration_anchor_hash",
    "rollover_id",
    "wal_predecessor_generation_id",
    "wal_predecessor_record_hash",
    "compaction",
)
_STARTUP_AGGREGATE_DOMAIN = (
    b"PROJECT-SUZKA:R07:STARTUP-PARTICIPANT-AGGREGATE:V1\x00"
)
_PARTICIPANT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:\.[a-z0-9]+)*$")


class EventLifecycle(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    PREPARED = "prepared"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERY_CLASSIFIED = "recovery_classified"
    CHECKPOINT = "checkpoint"
    RECOVERY_PREPARED = "recovery_prepared"
    RECOVERY_COMPLETED = "recovery_completed"
    TRANSACTION_PREPARED = "transaction_prepared"
    PARTICIPANT_FINALIZED = "participant_finalized"
    TRANSACTION_COMPLETED = "transaction_completed"
    TRANSACTION_RECONCILIATION_REQUIRED = "transaction_reconciliation_required"
    TRANSACTION_RECONCILED = "transaction_reconciled"
    PARTICIPANT_ABORTED = "participant_aborted"
    TRANSACTION_ABORT_REQUIRED = "transaction_abort_required"
    TRANSACTION_ABORTED = "transaction_aborted"
    STARTUP_RECONCILIATION_PREPARED = "startup_reconciliation_prepared"
    STARTUP_PARTICIPANT_RECONCILED = "startup_participant_reconciled"
    STARTUP_RECONCILIATION_COMPLETED = "startup_reconciliation_completed"
    PARTICIPANT_BASELINE_ESTABLISHED = "participant_baseline_established"
    CLEAR_PREPARED = "clear_prepared"
    CLEARED = "cleared"
    WAL_GENERATION_ROLLED_PREPARED = "wal_generation_rolled_prepared"
    WAL_GENERATION_ROLLED_COMPLETED = "wal_generation_rolled_completed"
    WAL_GENERATION_ROLLED_ABORTED = "wal_generation_rolled_aborted"
    # Old segments archived; declares the verifier state at the oldest
    # retained checkpoint (#381 D1 P0-1, D2 P1-2).
    COMPACTION = "compaction"


_ROLLOVER_LIFECYCLES = frozenset(
    {
        EventLifecycle.WAL_GENERATION_ROLLED_PREPARED,
        EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED,
        EventLifecycle.WAL_GENERATION_ROLLED_ABORTED,
    }
)
_ROLLOVER_FIELDS = (
    "rollover_id",
    "snapshot_sequence",
    "snapshot_hash",
    "wal_generation_id",
    "wal_record_id",
    "wal_record_hash",
    "wal_predecessor_generation_id",
    "wal_predecessor_record_hash",
    "journal_lineage_id",
)


class TransactionKind(str, Enum):
    EVENT_MUTATION = "event_mutation"
    MAINTENANCE_MUTATION = "maintenance_mutation"


class ParticipantOutcome(str, Enum):
    FINALIZED = "finalized"
    ALREADY_CONSISTENT = "already_consistent"


class ParticipantCapability(str, Enum):
    PREPARE = "prepare"
    ABORT = "abort"
    IDEMPOTENT_FINALIZE = "idempotent_finalize"
    INSPECT_RECONCILE = "inspect_reconcile"


class AbortOutcome(str, Enum):
    ABORTED = "aborted"
    ALREADY_ABSENT = "already_absent"


class ReconciliationReason(str, Enum):
    PARTICIPANT_FINALIZATION_INCOMPLETE = "participant_finalization_incomplete"
    PARTICIPANT_DIVERGED = "participant_diverged"
    PARTICIPANT_UNAVAILABLE = "participant_unavailable"
    UNSUPPORTED_RECONCILIATION = "unsupported_reconciliation"


class StartupParticipantOutcome(str, Enum):
    VERIFIED_CONSISTENT = "verified_consistent"
    ROLLED_FORWARD = "rolled_forward"


class EventFailureCategory(str, Enum):
    HANDLER_FAILURE = "handler_failure"
    ACCEPTED_NOT_STARTED = "accepted_not_started"
    UNCOMMITTED_AFTER_CRASH = "uncommitted_after_crash"
    COMMITTED_BEFORE_CRASH = "committed_before_crash"


class EventRecoveryCategory(str, Enum):
    EXACT_CURRENT = "exact_current"
    TRUE_ROLLBACK = "true_rollback"
    UNCOMMITTED_TAIL = "uncommitted_tail"


EventRecoveryClass = EventRecoveryCategory
RecoveryClass = EventRecoveryCategory


class ParticipantDomain(str, Enum):
    DURABLE_DOMAIN = "durable_domain"
    EPHEMERAL_PROCESS = "ephemeral_process"


class EventJournalAppendStage(str, Enum):
    VALIDATE = "validate"
    WRITE = "write"
    FILE_FSYNC = "file_fsync"
    PARENT_FSYNC = "parent_fsync"
    ROTATION = "rotation"


class _JournalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ParticipantBaseline(_JournalModel):
    participant_id: str = Field(min_length=1, max_length=64)
    domain: ParticipantDomain

    @field_validator("participant_id")
    @classmethod
    def validate_participant_id(cls, value: str) -> str:
        if _PARTICIPANT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("participant identifier is invalid")
        return value


class ParticipantRequirement(_JournalModel):
    participant_id: str = Field(min_length=1, max_length=64)
    operation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: tuple[ParticipantCapability, ...]

    @field_validator("participant_id")
    @classmethod
    def validate_participant_id(cls, value: str) -> str:
        if _PARTICIPANT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("participant identifier is invalid")
        return value

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(
        cls, value: tuple[ParticipantCapability, ...]
    ) -> tuple[ParticipantCapability, ...]:
        if (
            not value
            or value != tuple(sorted(value, key=lambda item: item.value))
            or len(set(value)) != len(value)
            or ParticipantCapability.PREPARE not in value
            or ParticipantCapability.IDEMPOTENT_FINALIZE not in value
        ):
            raise ValueError("participant capabilities are invalid")
        return value


class EventJournalCompaction(_JournalModel):
    """What a COMPACTION record declares (#381 D1/D2).

    The archived range is the oldest segments ending just before the
    boundary checkpoint. The embedded records are the minimal carried state
    that retained replay still needs; each is the original record verbatim,
    so its identity is proven by recomputing its own record hash.
    """

    compaction_id: str
    boundary_checkpoint_hash: str
    archived_first_record_hash: str
    archived_last_record_hash: str
    archived_record_count: int = Field(ge=1)
    archived_processing_floor: int = Field(ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE)
    archived_processing_high_water: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    archived_first_timestamp: datetime
    archived_last_timestamp: datetime
    archived_root: str
    archive_file: str
    archive_sha256: str
    archive_raw_sha256: str
    archive_bytes: int = Field(ge=1)
    retention_days: int = Field(ge=0, le=36_500)
    as_of: datetime
    previous_compaction_hash: str | None = None
    baseline_record: EventJournalRecord | None = None
    true_rollback_completed_record: EventJournalRecord | None = None
    true_rollback_cleared_record: EventJournalRecord | None = None
    carried_tail_record: EventJournalRecord | None = None

    @model_validator(mode="after")
    def validate_compaction(self) -> EventJournalCompaction:
        try:
            parsed = UUID(self.compaction_id)
        except (TypeError, ValueError, AttributeError):
            raise ValueError("compaction identifier is invalid") from None
        if str(parsed) != self.compaction_id:
            raise ValueError("compaction identifier is invalid")
        for value in (
            self.boundary_checkpoint_hash,
            self.archived_first_record_hash,
            self.archived_last_record_hash,
            self.archived_root,
            self.archive_sha256,
            self.archive_raw_sha256,
            self.previous_compaction_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("compaction hash is invalid")
        match = _ARCHIVE_NAME_PATTERN.fullmatch(self.archive_file)
        if (
            match is None
            or match.group(3) != self.compaction_id
            or int(match.group(1)) != self.archived_processing_floor
            or int(match.group(2)) != self.archived_processing_high_water
        ):
            raise ValueError("compaction archive name is invalid")
        if self.archived_processing_floor > self.archived_processing_high_water:
            raise ValueError("compaction range is invalid")
        for moment in (
            self.archived_first_timestamp,
            self.archived_last_timestamp,
            self.as_of,
        ):
            if moment.tzinfo is None:
                raise ValueError("compaction timestamps must be timezone-aware")
        if self.archived_first_timestamp > self.archived_last_timestamp:
            raise ValueError("compaction range is invalid")
        if (
            self.baseline_record is not None
            and self.baseline_record.lifecycle
            is not EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED
        ) or (
            self.true_rollback_completed_record is not None
            and self.true_rollback_completed_record.lifecycle
            is not EventLifecycle.RECOVERY_COMPLETED
        ) or (
            self.true_rollback_cleared_record is not None
            and self.true_rollback_cleared_record.lifecycle
            is not EventLifecycle.CLEARED
        ):
            raise ValueError("compaction carried record kind is invalid")
        if (self.true_rollback_completed_record is None) != (
            self.true_rollback_cleared_record is None
        ):
            raise ValueError("compaction true-rollback pair is incomplete")
        for carried in self.carried_records:
            if carried.lifecycle is EventLifecycle.COMPACTION:
                raise ValueError("compaction cannot carry a compaction")
        return self

    @property
    def carried_records(self) -> tuple[EventJournalRecord, ...]:
        return tuple(
            record
            for record in (
                self.baseline_record,
                self.true_rollback_completed_record,
                self.true_rollback_cleared_record,
                self.carried_tail_record,
            )
            if record is not None
        )


class EventJournalRecord(_JournalModel):
    schema_version: Literal[1, 2, 3, 4]
    record_id: str = Field(min_length=1)
    timestamp: datetime
    lifecycle: EventLifecycle
    event_id: str | None = None
    event_type: AgentEventType | None = None
    source: AgentEventSource | None = None
    processing_sequence: int | None = Field(
        default=None, ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    snapshot_sequence: int | None = Field(
        default=None, ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    snapshot_hash: str | None = None
    failure_category: EventFailureCategory | None = None
    previous_record_hash: str | None = None
    record_hash: str
    wal_generation_id: str | None = None
    wal_record_id: str | None = None
    wal_record_hash: str | None = None
    migration_previous_v1_hash: str | None = None
    journal_lineage_id: str | None = None
    recovery_id: str | None = None
    recovery_processing_high_water: int | None = Field(
        default=None, ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    recovery_category: EventRecoveryCategory | None = None
    external_reconciliation_required: bool | None = None
    migration_previous_v2_hash: str | None = None
    v3_migration_anchor_hash: str | None = None
    transaction_id: str | None = None
    transaction_kind: TransactionKind | None = None
    required_participants: tuple[ParticipantRequirement, ...] | None = None
    participant_id: str | None = None
    operation_digest: str | None = None
    participant_outcome: ParticipantOutcome | None = None
    abort_outcome: AbortOutcome | None = None
    abort_reason: ReconciliationReason | None = None
    reconciliation_id: str | None = None
    startup_participant_outcome: StartupParticipantOutcome | None = None
    reconciliation_reason: ReconciliationReason | None = None
    unresolved_participants: tuple[str, ...] | None = None
    baseline_id: str | None = None
    participant_registry: tuple[ParticipantBaseline, ...] | None = None
    adoption_epoch: int | None = Field(
        default=None, ge=0, le=MAX_PERSISTED_REVISION
    )
    migration_previous_v3_hash: str | None = None
    v4_migration_anchor_hash: str | None = None
    rollover_id: str | None = None
    wal_predecessor_generation_id: str | None = None
    wal_predecessor_record_hash: str | None = None
    compaction: EventJournalCompaction | None = None

    @field_validator(
        "processing_sequence",
        "snapshot_sequence",
        "recovery_processing_high_water",
        mode="before",
    )
    @classmethod
    def require_exact_persisted_sequence(cls, value: object) -> object:
        if value is not None and type(value) is not int:
            raise ValueError("persisted processing sequence must be an integer")
        return value

    @field_validator("record_id", "event_id")
    @classmethod
    def require_canonical_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = UUID(value)
        except ValueError:
            raise ValueError("identifier must be a UUID") from None
        if str(parsed) != value:
            raise ValueError("identifier must use canonical UUID form")
        return value

    @field_validator("participant_id")
    @classmethod
    def validate_record_participant_id(cls, value: str | None) -> str | None:
        if value is not None and _PARTICIPANT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("participant identifier is invalid")
        return value

    @model_validator(mode="after")
    def validate_record_shape(self) -> EventJournalRecord:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        for value in (
            self.state_hash_before,
            self.state_hash_after,
            self.snapshot_hash,
            self.previous_record_hash,
            self.record_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("hash must be lowercase SHA-256")

        identity = (self.event_id, self.event_type, self.source)
        has_identity = all(value is not None for value in identity)
        no_identity = all(value is None for value in identity)
        if not has_identity and not no_identity:
            raise ValueError("event identity must be complete")

        for value in (
            self.wal_generation_id,
            self.wal_record_id,
            self.recovery_id,
            self.reconciliation_id,
            self.journal_lineage_id,
            self.baseline_id,
            self.rollover_id,
            self.wal_predecessor_generation_id,
        ):
            if value is not None:
                try:
                    parsed = UUID(value)
                except ValueError:
                    raise ValueError("identifier must be a UUID") from None
                if str(parsed) != value:
                    raise ValueError("identifier must use canonical UUID form")
        if self.transaction_id is not None:
            try:
                parsed = UUID(self.transaction_id)
            except ValueError:
                raise ValueError("identifier must be a UUID") from None
            if str(parsed) != self.transaction_id:
                raise ValueError("identifier must use canonical UUID form")
        for value in (
            self.operation_digest,
            self.migration_previous_v2_hash,
            self.v3_migration_anchor_hash,
            self.migration_previous_v3_hash,
            self.v4_migration_anchor_hash,
            self.wal_predecessor_record_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("hash must be lowercase SHA-256")
        if self.required_participants is not None:
            ids = tuple(item.participant_id for item in self.required_participants)
            if not ids or ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
                raise ValueError("required participants must be unique and sorted")
        if self.unresolved_participants is not None:
            if (
                not self.unresolved_participants
                or any(
                    _PARTICIPANT_ID_PATTERN.fullmatch(x) is None
                    for x in self.unresolved_participants
                )
                or self.unresolved_participants
                != tuple(sorted(self.unresolved_participants))
                or len(set(self.unresolved_participants))
                != len(self.unresolved_participants)
            ):
                raise ValueError("unresolved participants must be unique and sorted")

        v3_fields = (
            self.migration_previous_v2_hash,
            self.v3_migration_anchor_hash,
            self.transaction_id,
            self.transaction_kind,
            self.required_participants,
            self.participant_id,
            self.operation_digest,
            self.participant_outcome,
            self.abort_outcome,
            self.abort_reason,
            self.reconciliation_reason,
            self.unresolved_participants,
            self.reconciliation_id,
            self.startup_participant_outcome,
            self.baseline_id,
            self.participant_registry,
            self.adoption_epoch,
        )
        if self.schema_version in {1, 2} and any(
            value is not None for value in v3_fields
        ):
            raise ValueError("v3 fields require schema 3")
        if self.schema_version < 4 and any(
            getattr(self, name) is not None for name in _V4_FIELDS
        ):
            raise ValueError("v4 fields require schema 4")
        if self.schema_version == 4 and (
            self.migration_previous_v1_hash is not None
            or self.migration_previous_v2_hash is not None
        ):
            raise ValueError("v4 records cannot carry older migrations")
        for value in (
            self.wal_record_hash,
            self.migration_previous_v1_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("hash must be lowercase SHA-256")

        if self.schema_version == 1 and any(
            value is not None
            for value in (
                self.wal_generation_id,
                self.wal_record_id,
                self.wal_record_hash,
                self.migration_previous_v1_hash,
                self.journal_lineage_id,
                self.recovery_id,
                self.recovery_processing_high_water,
                self.recovery_category,
                self.external_reconciliation_required,
            )
        ):
            raise ValueError("v1 has no v2 fields")
        transaction_lifecycles = {
            EventLifecycle.TRANSACTION_PREPARED,
            EventLifecycle.PARTICIPANT_FINALIZED,
            EventLifecycle.TRANSACTION_COMPLETED,
            EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
            EventLifecycle.TRANSACTION_RECONCILED,
            EventLifecycle.PARTICIPANT_ABORTED,
            EventLifecycle.TRANSACTION_ABORT_REQUIRED,
            EventLifecycle.TRANSACTION_ABORTED,
        }
        if self.lifecycle in transaction_lifecycles:
            if (
                self.schema_version < 3
                or not has_identity
                or self.processing_sequence is None
            ):
                raise ValueError("transaction record identity is incomplete")
            allowed_transaction_fields = {
                EventLifecycle.TRANSACTION_PREPARED: {
                    "transaction_id",
                    "transaction_kind",
                    "required_participants",
                },
                EventLifecycle.PARTICIPANT_FINALIZED: {
                    "transaction_id",
                    "participant_id",
                    "operation_digest",
                    "participant_outcome",
                },
                EventLifecycle.TRANSACTION_COMPLETED: {"transaction_id"},
                EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED: {
                    "transaction_id",
                    "reconciliation_reason",
                    "unresolved_participants",
                },
                EventLifecycle.TRANSACTION_RECONCILED: {"transaction_id"},
                EventLifecycle.PARTICIPANT_ABORTED: {
                    "transaction_id",
                    "participant_id",
                    "operation_digest",
                    "abort_outcome",
                },
                EventLifecycle.TRANSACTION_ABORT_REQUIRED: {
                    "transaction_id",
                    "abort_reason",
                    "unresolved_participants",
                },
                EventLifecycle.TRANSACTION_ABORTED: {"transaction_id"},
            }[self.lifecycle]
            self._forbid_irrelevant_fields(
                {"processing_sequence", *allowed_transaction_fields}
            )
            if self.lifecycle is EventLifecycle.TRANSACTION_PREPARED and (
                self.transaction_id is None
                or self.transaction_kind is None
                or not self.required_participants
            ):
                raise ValueError("transaction preparation is incomplete")
            if self.lifecycle is EventLifecycle.PARTICIPANT_FINALIZED and (
                self.transaction_id is None
                or self.participant_id is None
                or self.operation_digest is None
                or self.participant_outcome is None
            ):
                raise ValueError("participant finalization is incomplete")
            if self.lifecycle is EventLifecycle.PARTICIPANT_ABORTED and (
                self.transaction_id is None
                or self.participant_id is None
                or self.operation_digest is None
                or self.abort_outcome is None
            ):
                raise ValueError("participant abort is incomplete")
            if (
                self.lifecycle
                in {
                    EventLifecycle.TRANSACTION_COMPLETED,
                    EventLifecycle.TRANSACTION_RECONCILED,
                }
                and self.transaction_id is None
            ):
                raise ValueError("transaction completion is incomplete")
            if (
                self.lifecycle is EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED
                and (
                    self.transaction_id is None
                    or self.reconciliation_reason is None
                    or not self.unresolved_participants
                )
            ):
                raise ValueError("transaction reconciliation is incomplete")
            if self.lifecycle is EventLifecycle.TRANSACTION_ABORT_REQUIRED and (
                self.transaction_id is None
                or self.abort_reason is None
                or not self.unresolved_participants
            ):
                raise ValueError("transaction abort is incomplete")
            if (
                self.lifecycle is EventLifecycle.TRANSACTION_ABORTED
                and self.transaction_id is None
            ):
                raise ValueError("transaction abort completion is incomplete")
            return self
        startup_lifecycles = {
            EventLifecycle.STARTUP_RECONCILIATION_PREPARED,
            EventLifecycle.STARTUP_PARTICIPANT_RECONCILED,
            EventLifecycle.STARTUP_RECONCILIATION_COMPLETED,
        }
        if self.lifecycle in startup_lifecycles:
            if (
                self.schema_version < 3
                or not no_identity
                or self.processing_sequence is not None
                or self.reconciliation_id is None
                or self.recovery_id is None
                or self.snapshot_sequence is None
                or self.snapshot_hash is None
                or self.recovery_processing_high_water is None
                or self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
                or self.journal_lineage_id is None
            ):
                raise ValueError("startup reconciliation binding is incomplete")
            allowed = {
                EventLifecycle.STARTUP_RECONCILIATION_PREPARED: {
                    "reconciliation_id",
                    "recovery_id",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "recovery_processing_high_water",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "journal_lineage_id",
                    "required_participants",
                },
                EventLifecycle.STARTUP_PARTICIPANT_RECONCILED: {
                    "reconciliation_id",
                    "recovery_id",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "recovery_processing_high_water",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "journal_lineage_id",
                    "participant_id",
                    "operation_digest",
                    "startup_participant_outcome",
                },
                EventLifecycle.STARTUP_RECONCILIATION_COMPLETED: {
                    "reconciliation_id",
                    "recovery_id",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "recovery_processing_high_water",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "journal_lineage_id",
                },
            }[self.lifecycle]
            self._forbid_irrelevant_fields(allowed)
            if self.lifecycle is EventLifecycle.STARTUP_RECONCILIATION_PREPARED:
                if not self.required_participants:
                    raise ValueError("startup preparation participants are missing")
            elif self.lifecycle is EventLifecycle.STARTUP_PARTICIPANT_RECONCILED:
                if (
                    self.participant_id is None
                    or self.operation_digest is None
                    or self.startup_participant_outcome is None
                ):
                    raise ValueError("startup participant evidence is incomplete")
            return self
        if self.lifecycle is EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED:
            if (
                self.schema_version < 3
                or not no_identity
                or self.processing_sequence is not None
                or self.baseline_id is None
                or self.snapshot_sequence is None
                or self.snapshot_hash is None
                or self.recovery_processing_high_water is None
                or self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
                or self.journal_lineage_id is None
            ):
                raise ValueError("participant baseline binding is incomplete")
            allowed = {
                "baseline_id", "snapshot_sequence", "snapshot_hash",
                "recovery_processing_high_water",
                "wal_generation_id", "wal_record_id", "wal_record_hash",
                "journal_lineage_id", "participant_registry", "adoption_epoch",
            }
            if not self.participant_registry:
                raise ValueError("participant baseline registry is missing")
            self._forbid_irrelevant_fields(allowed)
            return self
        if self.lifecycle in {EventLifecycle.CLEAR_PREPARED, EventLifecycle.CLEARED}:
            if (
                self.schema_version < 3
                or not no_identity
                or self.processing_sequence is not None
                or self.baseline_id is None
                or self.recovery_id is None
                or self.reconciliation_id is None
                or self.snapshot_sequence is None
                or self.snapshot_hash is None
                or self.recovery_processing_high_water is None
                or self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
                or self.journal_lineage_id is None
            ):
                raise ValueError("external gate clear binding is incomplete")
            self._forbid_irrelevant_fields({
                "baseline_id", "recovery_id", "reconciliation_id",
                "snapshot_sequence", "snapshot_hash",
                "recovery_processing_high_water", "wal_generation_id",
                "wal_record_id", "wal_record_hash", "journal_lineage_id",
            })
            return self
        if self.lifecycle is EventLifecycle.COMPACTION:
            if (
                self.schema_version < 4
                or not no_identity
                or self.processing_sequence is not None
                or self.compaction is None
                or self.journal_lineage_id is None
            ):
                raise ValueError("compaction record is incomplete")
            self._forbid_irrelevant_fields({"compaction", "journal_lineage_id"})
            return self
        if self.lifecycle in _ROLLOVER_LIFECYCLES:
            if (
                self.schema_version < 4
                or not no_identity
                or self.processing_sequence is not None
                or any(getattr(self, name) is None for name in _ROLLOVER_FIELDS)
            ):
                raise ValueError("WAL rollover binding is incomplete")
            if self.wal_generation_id == self.wal_predecessor_generation_id:
                raise ValueError("WAL rollover must change generation")
            self._forbid_irrelevant_fields(set(_ROLLOVER_FIELDS))
            return self
        if self.lifecycle is EventLifecycle.CHECKPOINT:
            if not no_identity or self.processing_sequence is None:
                raise ValueError("checkpoint identity is invalid")
            self._require_snapshot()
            self._forbid_state_and_failure()
            if self.schema_version == 2 and self.journal_lineage_id is None:
                raise ValueError("v2 checkpoint lineage is missing")
            if self.schema_version >= 2 and self.journal_lineage_id is None:
                raise ValueError("checkpoint lineage is missing")
            if self.schema_version >= 2 and (
                self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
                or self.external_reconciliation_required is None
            ):
                raise ValueError("v2 checkpoint WAL identity is incomplete")
            if self.schema_version >= 3 and self.v3_migration_anchor_hash is None:
                raise ValueError("v3 checkpoint migration anchor is missing")
            if self.schema_version >= 4 and self.v4_migration_anchor_hash is None:
                raise ValueError("v4 checkpoint migration anchor is missing")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "migration_previous_v1_hash",
                    "migration_previous_v2_hash",
                    "v3_migration_anchor_hash",
                    "migration_previous_v3_hash",
                    "v4_migration_anchor_hash",
                    "journal_lineage_id",
                    "external_reconciliation_required",
                }
            )
        elif not has_identity and self.lifecycle not in {
            EventLifecycle.RECOVERY_PREPARED,
            EventLifecycle.RECOVERY_COMPLETED,
        }:
            raise ValueError("event lifecycle requires identity")
        elif self.lifecycle is EventLifecycle.ACCEPTED:
            self._require_only()
            self._forbid_irrelevant_fields(set())
        elif self.lifecycle is EventLifecycle.STARTED:
            if self.processing_sequence is None:
                raise ValueError("started requires processing sequence")
            self._forbid_state_snapshot_failure()
            self._forbid_irrelevant_fields({"processing_sequence"})
        elif self.lifecycle is EventLifecycle.PREPARED:
            if (
                self.processing_sequence is None
                or self.state_hash_before is None
                or self.state_hash_after is None
            ):
                raise ValueError("prepared requires sequence and state hashes")
            if any(
                value is not None
                for value in (
                    self.snapshot_sequence,
                    self.snapshot_hash,
                    self.failure_category,
                )
            ):
                raise ValueError("prepared has forbidden fields")
            if self.schema_version >= 2 and (self.wal_generation_id is None):
                raise ValueError("v2 prepared WAL identity is incomplete")
            if self.schema_version >= 2 and (
                self.wal_record_id is not None or self.wal_record_hash is not None
            ):
                raise ValueError("v2 prepared cannot bind appended WAL record")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "state_hash_before",
                    "state_hash_after",
                    "wal_generation_id",
                }
            )
        elif self.lifecycle is EventLifecycle.COMPLETED:
            if self.processing_sequence is None:
                raise ValueError("completed requires processing sequence")
            self._require_snapshot()
            self._forbid_state_and_failure()
            if self.schema_version >= 2 and (
                self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
            ):
                raise ValueError("v2 completed WAL identity is incomplete")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                }
            )
        elif self.lifecycle is EventLifecycle.FAILED:
            if self.processing_sequence is None:
                raise ValueError("failed requires processing sequence")
            self._require_snapshot()
            if self.failure_category is not EventFailureCategory.HANDLER_FAILURE:
                raise ValueError("failed category is invalid")
            self._forbid_state()
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "failure_category",
                }
            )
        elif self.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED:
            self._require_snapshot()
            self._forbid_state()
            if self.failure_category is EventFailureCategory.ACCEPTED_NOT_STARTED:
                if self.processing_sequence is not None:
                    raise ValueError("accepted-only recovery cannot consume sequence")
            elif self.failure_category not in {
                EventFailureCategory.UNCOMMITTED_AFTER_CRASH,
                EventFailureCategory.COMMITTED_BEFORE_CRASH,
            }:
                raise ValueError("recovery category is invalid")
            elif self.processing_sequence is None:
                raise ValueError("processing recovery requires sequence")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "failure_category",
                }
            )
        elif self.lifecycle is EventLifecycle.RECOVERY_PREPARED:
            if self.schema_version < 2 or self.recovery_id is None or not no_identity:
                raise ValueError("v2 recovery preparation is invalid")
            self._require_snapshot()
            if self.recovery_category is None or self.wal_generation_id is None:
                raise ValueError("v2 recovery preparation is incomplete")
            if self.recovery_processing_high_water is None:
                raise ValueError("recovery preparation high-water is missing")
            if self.external_reconciliation_required is not None:
                raise ValueError("recovery preparation has forbidden fields")
            self._forbid_state()
            self._forbid_irrelevant_fields(
                {
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "recovery_id",
                    "recovery_processing_high_water",
                    "recovery_category",
                }
            )
        elif self.lifecycle is EventLifecycle.RECOVERY_COMPLETED:
            if self.schema_version < 2 or self.recovery_id is None or not no_identity:
                raise ValueError("v2 recovery completion is invalid")
            self._require_snapshot()
            if self.recovery_category is None or self.wal_generation_id is None:
                raise ValueError("v2 recovery completion is incomplete")
            if self.recovery_processing_high_water is None:
                raise ValueError("recovery completion high-water is missing")
            if self.external_reconciliation_required is None:
                raise ValueError("recovery completion requires reconciliation flag")
            if self.wal_record_id is None or self.wal_record_hash is None:
                raise ValueError("recovery completion WAL identity is incomplete")
            self._forbid_state()
            self._forbid_irrelevant_fields(
                {
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "recovery_id",
                    "recovery_processing_high_water",
                    "recovery_category",
                    "external_reconciliation_required",
                    "wal_record_id",
                    "wal_record_hash",
                }
            )
        return self

    def _require_only(self) -> None:
        if any(
            value is not None
            for value in (
                self.processing_sequence,
                self.state_hash_before,
                self.state_hash_after,
                self.snapshot_sequence,
                self.snapshot_hash,
                self.failure_category,
            )
        ):
            raise ValueError("accepted has forbidden fields")

    def _require_snapshot(self) -> None:
        if self.snapshot_sequence is None or self.snapshot_hash is None:
            raise ValueError("snapshot identity is required")

    def _forbid_state(self) -> None:
        if self.state_hash_before is not None or self.state_hash_after is not None:
            raise ValueError("state hashes are forbidden")

    def _forbid_irrelevant_fields(self, allowed: set[str]) -> None:
        allowed = allowed | {"event_id", "event_type", "source"}
        fields = (
            "event_id",
            "event_type",
            "source",
            "processing_sequence",
            "state_hash_before",
            "state_hash_after",
            "snapshot_sequence",
            "snapshot_hash",
            "failure_category",
            "wal_generation_id",
            "wal_record_id",
            "wal_record_hash",
            "migration_previous_v1_hash",
            "journal_lineage_id",
            "recovery_id",
            "recovery_processing_high_water",
            "recovery_category",
            "external_reconciliation_required",
            "migration_previous_v2_hash",
            "v3_migration_anchor_hash",
            "transaction_id",
            "transaction_kind",
            "required_participants",
            "participant_id",
            "operation_digest",
            "participant_outcome",
            "abort_outcome",
            "abort_reason",
            "reconciliation_id",
            "startup_participant_outcome",
            "reconciliation_reason",
            "unresolved_participants",
            "baseline_id",
            "participant_registry",
            "adoption_epoch",
            *_V4_FIELDS,
        )
        if any(
            getattr(self, field) is not None for field in fields if field not in allowed
        ):
            raise ValueError(f"{self.lifecycle.value} has forbidden fields")

    def _forbid_state_and_failure(self) -> None:
        self._forbid_state()
        if self.failure_category is not None:
            raise ValueError("failure category is forbidden")

    def _forbid_state_snapshot_failure(self) -> None:
        self._forbid_state_and_failure()
        if self.snapshot_sequence is not None or self.snapshot_hash is not None:
            raise ValueError("snapshot identity is forbidden")


class EventJournalRecovery(_JournalModel):
    processing_high_water: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    snapshot_sequence: int = Field(ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("processing_high_water", "snapshot_sequence", mode="before")
    @classmethod
    def require_exact_persisted_sequence(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("persisted processing sequence must be an integer")
        return value


class EventJournalError(Exception):
    """Base class for bounded Journal failures."""


class EventJournalLoadError(EventJournalError):
    """Journal artifacts cannot be loaded safely."""


class EventJournalIntegrityError(EventJournalLoadError):
    """Journal evidence is internally inconsistent."""


class UnsupportedEventJournalVersion(EventJournalLoadError):
    """A record uses an unsupported schema version."""


class EventJournalAppendError(EventJournalError):
    """A lifecycle record did not reach confirmed durable success."""

    def __init__(self, stage: EventJournalAppendStage, *, published: bool) -> None:
        self.stage = stage
        self.published = published
        super().__init__(
            "EventJournal append failed "
            f"at {stage.value}; published={str(published).lower()}"
        )


EventJournalCompaction.model_rebuild()


class EventJournalAdmissionBand(str, Enum):
    """Capacity band of the live Journal (#381 §6, D2 P1-4; IP-4b).

    Live bytes are the active segment plus the rotated segments; archives
    are not counted. ``SOFT`` admits only ``SOFT_BAND_ADMITTED_EVENT_TYPES``;
    ``HARD`` admits no new event. Neither band ever blocks the lifecycle of
    an already admitted event, recovery, reconciliation, rollover or
    COMPACTION: only new ``ACCEPTED`` records are gated.
    """

    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"


# #381 D2 P1-4: the soft band is an allow-list over all 16 AgentEventTypes.
# Cognition continues; DEBUG_CHAT, ADAPTER_EVALUATE and ADAPTER_UPDATE (Human
# decision 2026-11-10) are refused, and so is every type not listed here,
# including any type added later (R18/R19 external effects).
SOFT_BAND_ADMITTED_EVENT_TYPES: frozenset[AgentEventType] = frozenset(
    {
        AgentEventType.CHAT,
        AgentEventType.SLEEP,
        AgentEventType.CONTEXT_UPDATE,
        AgentEventType.EMOTION_TICK,
        AgentEventType.VALUE_GOVERNANCE,
        AgentEventType.NARRATIVE_REVIEW,
        AgentEventType.SELF_ATTEMPT_REVIEW,
        AgentEventType.MOTIVATION_SOURCE_REVIEW,
        AgentEventType.GOAL_PROPOSAL_REVIEW,
        AgentEventType.GOAL_ADOPTION_REVIEW,
        AgentEventType.COMMITMENT_PROPOSAL_REVIEW,
        AgentEventType.COMMITMENT_ACCEPTANCE_REVIEW,
        AgentEventType.PLAN_PREVIEW_REVIEW,
    }
)


@dataclass(frozen=True, slots=True)
class EventJournalAdmissionStatus:
    """Bounded, read-only evidence for mutation admission.

    ``available`` is False only when no new event may be admitted at all
    (hard limit, exhausted sequence, or an impossible rotation). In the soft
    band it stays True with ``reason="journal_soft_limit"``; whether a given
    event type is admitted is ``admits()``.
    """

    available: bool
    reason: str | None
    active_file_bytes: int
    max_bytes: int
    rotated_segment_count: int
    retained_files: int
    safe_rotation_possible: bool
    lifecycle_blocks_rotation: bool
    proof_retention_blocks_safe_pruning: bool
    band: EventJournalAdmissionBand = EventJournalAdmissionBand.NORMAL
    live_bytes: int = 0
    max_total_bytes: int | None = None
    soft_limit_bytes: int | None = None

    def admits(self, event_type: AgentEventType) -> bool:
        if not self.available:
            return False
        if self.band is EventJournalAdmissionBand.SOFT:
            return event_type in SOFT_BAND_ADMITTED_EVENT_TYPES
        return self.band is EventJournalAdmissionBand.NORMAL


@dataclass(frozen=True, slots=True)
class _EventState:
    event_id: str
    event_type: AgentEventType
    source: AgentEventSource
    requested_at: datetime
    lifecycle: EventLifecycle
    processing_sequence: int | None
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    wal_generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventJournalTransaction:
    transaction_id: str
    event_id: str
    event_type: AgentEventType
    source: AgentEventSource
    processing_sequence: int
    kind: TransactionKind
    required_participants: tuple[ParticipantRequirement, ...]
    participant_outcomes: tuple[tuple[str, ParticipantOutcome], ...]
    abort_outcomes: tuple[tuple[str, AbortOutcome], ...] = ()
    reconciliation_reason: ReconciliationReason | None = None
    abort_reason: ReconciliationReason | None = None
    unresolved_participants: tuple[str, ...] = ()
    terminal_lifecycle: EventLifecycle | None = None

    @property
    def known_outcomes(self) -> tuple[tuple[str, ParticipantOutcome], ...]:
        return self.participant_outcomes


def startup_participant_aggregate_digest(
    transactions: tuple[EventJournalTransaction, ...], participant_id: str
) -> str:
    """Commit to a participant's complete transaction set without payloads."""

    rows = sorted(
        [
            transaction.transaction_id,
            transaction.processing_sequence,
            transaction.kind.value,
            transaction.terminal_lifecycle.value
            if transaction.terminal_lifecycle is not None
            else None,
            requirement.participant_id,
            requirement.operation_digest,
        ]
        for transaction in transactions
        for requirement in transaction.required_participants
        if requirement.participant_id == participant_id
    )
    canonical = json.dumps(
        rows,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_STARTUP_AGGREGATE_DOMAIN + canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class EventJournalStartupReconciliation:
    reconciliation_id: str
    recovery_id: str
    snapshot_sequence: int
    snapshot_hash: str
    recovery_processing_high_water: int
    wal_generation_id: str
    wal_record_id: str
    wal_record_hash: str
    journal_lineage_id: str
    required_participants: tuple[ParticipantRequirement, ...]
    participant_outcomes: tuple[tuple[str, str, StartupParticipantOutcome], ...]
    completed: bool

    @property
    def known_outcomes(
        self,
    ) -> tuple[tuple[str, str, StartupParticipantOutcome], ...]:
        return self.participant_outcomes


@dataclass(frozen=True, slots=True)
class EventJournalParticipantBaseline:
    baseline_id: str
    snapshot_sequence: int
    snapshot_hash: str
    processing_high_water: int
    wal_generation_id: str
    wal_record_id: str
    wal_record_hash: str
    journal_lineage_id: str
    participant_registry: tuple[ParticipantBaseline, ...]
    adoption_epoch: int = 0

    @property
    def registry(self) -> tuple[ParticipantBaseline, ...]:
        return self.participant_registry


@dataclass(frozen=True, slots=True)
class EventJournalGateClear:
    baseline_id: str
    reconciliation_id: str
    recovery_id: str
    snapshot_sequence: int
    snapshot_hash: str
    processing_high_water: int
    wal_generation_id: str
    wal_record_id: str
    wal_record_hash: str
    journal_lineage_id: str
    terminal: bool


@dataclass(frozen=True, slots=True)
class _CachedSegment:
    """In-process record of a parsed, hash-verified segment prefix (not durable)."""

    length: int
    last_line: bytes
    records: tuple[EventJournalRecord, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedJournal:
    records: tuple[EventJournalRecord, ...]
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    open_events: tuple[_EventState, ...]
    journal_lineage_id: str | None
    external_reconciliation_required: bool
    wal_snapshot_sequence: int | None
    wal_snapshot_hash: str | None
    open_recoveries: tuple[EventJournalRecord, ...]
    wal_generation_id: str | None
    wal_record_id: str | None
    wal_record_hash: str | None
    transactions: tuple[EventJournalTransaction, ...] = ()
    startup_reconciliations: tuple[EventJournalStartupReconciliation, ...] = ()
    baselines: tuple[EventJournalParticipantBaseline, ...] = ()
    open_gate_clear: EventJournalGateClear | None = None
    terminal_gate_clear: EventJournalGateClear | None = None
    open_wal_rollover: EventJournalWalRollover | None = None
    latest_compaction: EventJournalCompaction | None = None
    carried_records: tuple[EventJournalRecord, ...] = ()


@dataclass(slots=True)
class _VerifierState:
    """Resumable state of ``_verify_records`` after a verified record prefix.

    ``_verify_records`` is a pure function of the record sequence. This holds
    its loop state after ``records`` so that a later call whose input starts
    with exactly these records only processes the new suffix (IP-2b, #383).
    It is in-process only and is owned by exactly one verification at a time.
    """

    records: tuple[EventJournalRecord, ...]
    high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    journal_lineage_id: str | None
    wal_generation_id: str | None
    wal_record_id: str | None
    wal_record_hash: str | None
    wal_snapshot_sequence: int | None
    wal_snapshot_hash: str | None
    external_reconciliation_required: bool
    v2_seen: bool
    v3_seen: bool
    v3_migration_anchor_hash: str | None
    v4_seen: bool = False
    v4_migration_anchor_hash: str | None = None
    rollover_open: EventJournalRecord | None = None
    last_compaction_hash: str | None = None
    injected_compaction: EventJournalRecord | None = None
    injected_compaction_passed: bool = False
    processing_event_id: str | None = None
    gate_clear_open: bool = False
    open_events: dict[str, _EventState] = field(default_factory=dict)
    accepted_queue: list[str] = field(default_factory=list)
    seen_event_ids: set[str] = field(default_factory=set)
    seen_record_ids: set[str] = field(default_factory=set)
    recovery_open: dict[str, EventJournalRecord] = field(default_factory=dict)
    seen_recovery_ids: set[str] = field(default_factory=set)
    completed_recoveries: dict[str, EventJournalRecord] = field(default_factory=dict)
    startup_open: dict[str, dict[str, Any]] = field(default_factory=dict)
    startup_seen_ids: set[str] = field(default_factory=set)
    startup_seen_recovery_ids: set[str] = field(default_factory=set)
    startup_completed: list[dict[str, Any]] = field(default_factory=list)
    seen_rollover_ids: set[str] = field(default_factory=set)
    transactions: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline_records_seen: list[EventJournalRecord] = field(default_factory=list)
    gate_records: list[EventJournalRecord] = field(default_factory=list)
    transaction_views: dict[str, EventJournalTransaction] = field(
        default_factory=dict
    )
    result: _VerifiedJournal | None = None


# Records that bind the Journal to the current StateWAL anchor. A completed
# generation rollover moves that anchor to the new generation's baseline.
_WAL_ANCHOR_LIFECYCLES = frozenset(
    {
        EventLifecycle.CHECKPOINT,
        EventLifecycle.COMPLETED,
        EventLifecycle.RECOVERY_COMPLETED,
        EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED,
    }
)


_GATE_LIFECYCLES = frozenset(
    {
        EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED,
        EventLifecycle.CLEAR_PREPARED,
        EventLifecycle.CLEARED,
    }
)


@dataclass(frozen=True, slots=True)
class EventJournalCompactionOutcome:
    """Result of one compaction attempt."""

    performed: bool
    reason: str | None
    archived_segments: int = 0
    archived_records: int = 0
    archive_file: str | None = None


@dataclass(frozen=True, slots=True)
class EventJournalArchiveAlert:
    """A referenced archive that is missing or does not match (#381 D2 P2).

    Archives are not authority: an alert is reported, never a fail-stop.
    """

    archive_file: str
    reason: str


@dataclass(frozen=True, slots=True)
class EventJournalWalRollover:
    """A prepared, not yet completed or aborted, StateWAL generation rollover."""

    rollover_id: str
    snapshot_sequence: int
    snapshot_hash: str
    wal_generation_id: str
    wal_record_id: str
    wal_record_hash: str
    wal_predecessor_generation_id: str
    wal_predecessor_record_hash: str


@dataclass(frozen=True, slots=True)
class EventJournalWalRolloverBoundary:
    """Evidence for deciding a rollover while the Journal lock is held."""

    clean: bool
    reason: str | None
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    wal_generation_id: str | None
    wal_record_id: str | None
    wal_record_hash: str | None


@dataclass(frozen=True, slots=True)
class EventJournalOpenEvent:
    event_id: str
    lifecycle: EventLifecycle
    processing_sequence: int | None
    classification: str


@dataclass(frozen=True, slots=True)
class EventJournalOpenRecovery:
    recovery_id: str
    category: EventRecoveryCategory
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    wal_generation_id: str


@dataclass(frozen=True, slots=True)
class EventJournalInspection:
    """Read-only, bounded evidence for a parent authority to evaluate."""

    records: tuple[EventJournalRecord, ...]
    tail_record_id: str | None
    tail_record_hash: str | None
    schema_version: int | None
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    open_events: tuple[EventJournalOpenEvent, ...]
    classification_plan: tuple[EventJournalOpenEvent, ...]
    open_recoveries: tuple[EventJournalOpenRecovery, ...]
    sequence_evidence: tuple[int, ...]
    journal_lineage_id: str | None
    external_reconciliation_required: bool
    wal_snapshot_sequence: int | None
    wal_snapshot_hash: str | None
    wal_generation_id: str | None
    wal_record_id: str | None
    wal_record_hash: str | None
    open_transactions: tuple[EventJournalTransaction, ...] = ()
    reconciliation_required_transactions: tuple[EventJournalTransaction, ...] = ()
    completed_transactions: tuple[EventJournalTransaction, ...] = ()
    reconciled_transactions: tuple[EventJournalTransaction, ...] = ()
    abort_required_transactions: tuple[EventJournalTransaction, ...] = ()
    aborted_transactions: tuple[EventJournalTransaction, ...] = ()
    open_startup_reconciliations: tuple[EventJournalStartupReconciliation, ...] = ()
    completed_startup_reconciliations: tuple[
        EventJournalStartupReconciliation, ...
    ] = ()
    baselines: tuple[EventJournalParticipantBaseline, ...] = ()
    open_gate_clear: EventJournalGateClear | None = None
    terminal_gate_clear: EventJournalGateClear | None = None
    open_wal_rollover: EventJournalWalRollover | None = None
    # Declared by the latest COMPACTION (#381): the archived range and the
    # original records carried across it (baseline, true-rollback pair,
    # Semantic checkpoint tail). They are not part of ``records``.
    latest_compaction: EventJournalCompaction | None = None
    carried_records: tuple[EventJournalRecord, ...] = ()

    @property
    def participant_baselines(self) -> tuple[EventJournalParticipantBaseline, ...]:
        return self.baselines

    @property
    def gate_clear(self) -> EventJournalGateClear | None:
        return self.terminal_gate_clear or self.open_gate_clear


class EventJournalLease:
    """Exclusive process lease protecting Journal and Snapshot startup authority."""

    def __init__(self, journal_path: str | Path) -> None:
        self.path = Path(journal_path)
        self._descriptor: int | None = None
        lease_failure: EventJournalLoadError | None = None
        descriptor: int | None = None
        parent_descriptor: int | None = None
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_descriptor = os.open(
                self.path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            parent_status = os.fstat(parent_descriptor)
            if (
                not stat.S_ISDIR(parent_status.st_mode)
                or parent_status.st_uid != os.geteuid()
            ):
                raise OSError("journal directory is not private")
            if parent_status.st_mode & 0o077:
                os.fchmod(parent_descriptor, 0o700)
                if os.fstat(parent_descriptor).st_mode & 0o077:
                    raise OSError("journal directory hardening failed")
            descriptor = os.open(
                f".{self.path.name}.lock",
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            lock_status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lock_status.st_mode)
                or lock_status.st_uid != os.geteuid()
            ):
                raise OSError("journal lock is not regular")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._descriptor = descriptor
            descriptor = None
            os.fsync(parent_descriptor)
        except OSError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            lease_failure = EventJournalLoadError(
                "EventJournal exclusive authority is unavailable"
            )
        finally:
            if parent_descriptor is not None:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass
        if lease_failure is not None:
            raise lease_failure

    @property
    def held(self) -> bool:
        return self._descriptor is not None

    def close(self) -> None:
        if self._descriptor is not None:
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)
                self._descriptor = None

    def _transfer(self) -> EventJournalLease:
        if self._descriptor is None:
            raise ValueError("EventJournal lease is not held")
        adopted = object.__new__(EventJournalLease)
        adopted.path = self.path
        adopted._descriptor = self._descriptor
        self._descriptor = None
        return adopted

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _fsync_parent(self) -> None:
        descriptor = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class EventJournal:
    """Append and verify durable lifecycle records without event payloads."""

    def __init__(
        self,
        path: str | Path,
        max_bytes: int,
        retained_files: int,
        *,
        clock: Callable[[], datetime] | None = None,
        append_stage_hook: (
            Callable[[EventLifecycle, EventJournalAppendStage], None] | None
        ) = None,
        lease: EventJournalLease | None = None,
        archive_dir: str | Path | None = None,
        max_total_bytes: int | None = None,
        soft_band_bytes: int = 0,
    ) -> None:
        if max_total_bytes is not None and (
            isinstance(max_total_bytes, bool)
            or not isinstance(max_total_bytes, int)
            or max_total_bytes <= 0
        ):
            raise ValueError("max_total_bytes must be a positive integer")
        if (
            isinstance(soft_band_bytes, bool)
            or not isinstance(soft_band_bytes, int)
            or soft_band_bytes < 0
            or (max_total_bytes is None and soft_band_bytes != 0)
            or (max_total_bytes is not None and soft_band_bytes >= max_total_bytes)
        ):
            raise ValueError(
                "soft_band_bytes must be non-negative, below max_total_bytes, "
                "and requires max_total_bytes"
            )
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(retained_files, bool)
            or not isinstance(retained_files, int)
            or retained_files < 2
        ):
            raise ValueError("retained_files must be at least two")
        self.path = Path(path)
        self.max_bytes = max_bytes
        # Deprecated: retention is by time and compaction (#381 D2 P2). Kept
        # for configuration compatibility and admission status only.
        self.retained_files = retained_files
        # IP-4b: live bytes (active + rotated, never archives). None keeps
        # the pre-IP-4b behaviour of no total limit.
        self.max_total_bytes = max_total_bytes
        self.soft_band_bytes = soft_band_bytes
        self._last_band = EventJournalAdmissionBand.NORMAL
        # Outside the rotated-segment glob ``<name>.<8 digits>``.
        self.archive_dir = (
            Path(archive_dir)
            if archive_dir is not None
            else self.path.with_name(f"{self.path.name}-archive")
        )
        self.archive_alerts: tuple[EventJournalArchiveAlert, ...] = ()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._append_stage_hook = append_stage_hook
        self._lock = RLock()
        self._segment_cache: dict[tuple[int, int], _CachedSegment] = {}
        self._verifier_state: _VerifierState | None = None
        if lease is not None and (lease.path != self.path or not lease.held):
            raise ValueError("EventJournal lease does not match configured path")
        self._lease = (
            lease._transfer() if lease is not None else EventJournalLease(self.path)
        )
        with self._lock:
            try:
                self._roll_forward_compaction_unlocked()
                records = self._read_records_unlocked()
                if records:
                    self._verify_records(records)
                    self.archive_alerts = (
                        *self.archive_alerts,
                        *self._check_archives_unlocked(records),
                    )
                    for alert in self.archive_alerts:
                        LOGGER.warning(
                            "EventJournal archive alert: %s %s",
                            alert.archive_file,
                            alert.reason,
                        )
                self._quarantine_unreferenced_archives_unlocked(records)
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Release this process's exclusive journal authority."""

        with self._lock:
            self._lease.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def records(self) -> tuple[EventJournalRecord, ...]:
        with self._lock:
            self._require_authority()
            return self._read_records_unlocked()

    @property
    def has_exclusive_authority(self) -> bool:
        return self._lease.held

    @property
    def has_history(self) -> bool:
        with self._lock:
            self._require_authority()
            return bool(self._read_records_unlocked())

    def admission_status(self) -> EventJournalAdmissionStatus:
        """Inspect bounded retention admission without mutating the Journal."""
        with self._lock:
            self._require_authority()
            return self._admission_status_unlocked()

    def append_accepted_if_admission_available(self, event: AgentEvent) -> bool:
        """Atomically guard new admission and durably append ``ACCEPTED``."""

        with self._lock:
            self._require_authority()
            if not self._admission_status_unlocked().admits(event.event_type):
                return False
            self.append_accepted(event)
            return True

    def inspect(
        self,
        snapshot_sequence: int | None = None,
        snapshot_hash: str | None = None,
    ) -> EventJournalInspection:
        """Return verified evidence without appending, rotating, or migrating."""
        if snapshot_sequence is not None and (
            type(snapshot_sequence) is not int
            or snapshot_sequence < 0
            or snapshot_sequence > MAX_PERSISTED_EVENT_SEQUENCE
        ):
            raise ValueError("canonical snapshot sequence is outside its bound")
        if snapshot_sequence is not None and snapshot_hash is not None:
            self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                return EventJournalInspection(
                    records=(),
                    tail_record_id=None,
                    tail_record_hash=None,
                    schema_version=None,
                    processing_high_water=0,
                    snapshot_sequence=0,
                    snapshot_hash="0" * 64,
                    open_events=(),
                    classification_plan=(),
                    open_recoveries=(),
                    sequence_evidence=(),
                    journal_lineage_id=None,
                    external_reconciliation_required=False,
                    wal_snapshot_sequence=None,
                    wal_snapshot_hash=None,
                    wal_generation_id=None,
                    wal_record_id=None,
                    wal_record_hash=None,
                    open_transactions=(),
                    reconciliation_required_transactions=(),
                    completed_transactions=(),
                    reconciled_transactions=(),
                    abort_required_transactions=(),
                    aborted_transactions=(),
                    open_startup_reconciliations=(),
                    completed_startup_reconciliations=(),
                    baselines=(),
                    open_gate_clear=None,
                    terminal_gate_clear=None,
                )
            verified = self._verify_records(records)
            classification_sequence = (
                verified.snapshot_sequence
                if snapshot_sequence is None
                else snapshot_sequence
            )
            classification_hash = (
                verified.snapshot_hash if snapshot_hash is None else snapshot_hash
            )
            self._validate_snapshot_identity(
                classification_sequence, classification_hash
            )
            open_events = self._plan_reconciliation(
                verified, classification_sequence, classification_hash
            )
            return EventJournalInspection(
                records=records,
                tail_record_id=records[-1].record_id,
                tail_record_hash=records[-1].record_hash,
                schema_version=records[-1].schema_version,
                processing_high_water=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                open_events=open_events,
                classification_plan=open_events,
                open_recoveries=tuple(
                    EventJournalOpenRecovery(
                        recovery_id=record.recovery_id or "",
                        category=record.recovery_category
                        or EventRecoveryCategory.EXACT_CURRENT,
                        processing_high_water=record.recovery_processing_high_water
                        or 0,
                        snapshot_sequence=record.snapshot_sequence or 0,
                        snapshot_hash=record.snapshot_hash or "0" * 64,
                        wal_generation_id=record.wal_generation_id or "",
                    )
                    for record in verified.open_recoveries
                ),
                sequence_evidence=tuple(
                    record.processing_sequence
                    for record in records
                    if record.processing_sequence is not None
                ),
                journal_lineage_id=verified.journal_lineage_id,
                external_reconciliation_required=(
                    verified.external_reconciliation_required
                ),
                wal_snapshot_sequence=verified.wal_snapshot_sequence,
                wal_snapshot_hash=verified.wal_snapshot_hash,
                wal_generation_id=verified.wal_generation_id,
                wal_record_id=verified.wal_record_id,
                wal_record_hash=verified.wal_record_hash,
                open_transactions=tuple(
                    t for t in verified.transactions if t.terminal_lifecycle is None
                ),
                reconciliation_required_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is None
                    and t.reconciliation_reason is not None
                ),
                completed_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is EventLifecycle.TRANSACTION_COMPLETED
                ),
                reconciled_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is EventLifecycle.TRANSACTION_RECONCILED
                ),
                abort_required_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.abort_reason is not None and t.terminal_lifecycle is None
                ),
                aborted_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is EventLifecycle.TRANSACTION_ABORTED
                ),
                open_startup_reconciliations=tuple(
                    item
                    for item in verified.startup_reconciliations
                    if not item.completed
                ),
                completed_startup_reconciliations=tuple(
                    item for item in verified.startup_reconciliations if item.completed
                ),
                baselines=verified.baselines,
                open_gate_clear=verified.open_gate_clear,
                terminal_gate_clear=verified.terminal_gate_clear,
                open_wal_rollover=verified.open_wal_rollover,
                latest_compaction=verified.latest_compaction,
                carried_records=verified.carried_records,
            )

    def apply_planned_reconciliation(
        self, snapshot_sequence: int, snapshot_hash: str
    ) -> EventJournalRecovery:
        """Apply the legacy reconciliation plan; migration remains explicit."""
        return self.verify_and_reconcile(snapshot_sequence, snapshot_hash)

    @staticmethod
    def _plan_reconciliation(
        verified: _VerifiedJournal,
        snapshot_sequence: int,
        snapshot_hash: str,
    ) -> tuple[EventJournalOpenEvent, ...]:
        """Validate a candidate identity and return a read-only crash plan."""

        processing = tuple(
            event
            for event in verified.open_events
            if event.lifecycle in {EventLifecycle.STARTED, EventLifecycle.PREPARED}
        )
        if len(processing) > 1:
            raise EventJournalIntegrityError(
                "EventJournal has multiple interrupted handlers"
            )
        canonical_matches = (
            snapshot_sequence == verified.snapshot_sequence
            and snapshot_hash == verified.snapshot_hash
        )
        committed_event_id: str | None = None
        if processing:
            interrupted = processing[0]
            if interrupted.lifecycle is EventLifecycle.PREPARED:
                if (
                    snapshot_sequence == interrupted.processing_sequence
                    and snapshot_hash == interrupted.state_hash_after
                ):
                    committed_event_id = interrupted.event_id
                elif not (
                    canonical_matches and snapshot_hash == interrupted.state_hash_before
                ):
                    raise EventJournalIntegrityError(
                        "Prepared transition does not match canonical snapshot"
                    )
            elif not canonical_matches:
                raise EventJournalIntegrityError(
                    "Started event does not match canonical snapshot"
                )
        elif not canonical_matches:
            raise EventJournalIntegrityError(
                "Journal and canonical snapshot are inconsistent"
            )
        return tuple(
            EventJournalOpenEvent(
                event_id=item.event_id,
                lifecycle=item.lifecycle,
                processing_sequence=item.processing_sequence,
                classification=(
                    "accepted_only"
                    if item.lifecycle is EventLifecycle.ACCEPTED
                    else "prepared_committed_before_crash"
                    if item.event_id == committed_event_id
                    else "started_or_prepared_uncommitted"
                ),
            )
            for item in verified.open_events
        )

    def append_v2_migration_checkpoint(
        self,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
    ) -> None:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                raise EventJournalIntegrityError("migration requires v1 history")
            verified = self._verify_records(records)
            if any(record.schema_version == 2 for record in records):
                raise EventJournalIntegrityError("EventJournal is already v2")
            if (
                verified.snapshot_sequence != snapshot_sequence
                or verified.snapshot_hash != snapshot_hash
            ):
                raise EventJournalIntegrityError("migration snapshot is inconsistent")
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=records[-1].record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
                schema_version=2,
                migration_previous_v1_hash=records[-1].record_hash,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=str(uuid4()),
                external_reconciliation_required=False,
            )
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    # Short name retained for callers that model this as a state transition.
    migrate_v2 = append_v2_migration_checkpoint

    def append_v3_migration_checkpoint(self) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) != 2:
                raise EventJournalIntegrityError("v3 migration requires v2 history")
            verified = self._verify_records(records)
            if (
                verified.open_events
                or verified.open_recoveries
                or verified.transactions
                or any(not item.completed for item in verified.startup_reconciliations)
            ):
                raise EventJournalIntegrityError(
                    "v3 migration requires a closed v2 journal"
                )
            prior = records[-1]
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=prior.record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                schema_version=3,
                migration_previous_v2_hash=prior.record_hash,
                v3_migration_anchor_hash=prior.record_hash,
                wal_generation_id=verified.wal_generation_id,
                wal_record_id=verified.wal_record_id,
                wal_record_hash=verified.wal_record_hash,
                journal_lineage_id=verified.journal_lineage_id,
                external_reconciliation_required=verified.external_reconciliation_required,
            )
            self._verify_records((*records, checkpoint))
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    migrate_v3 = append_v3_migration_checkpoint

    def append_v4_migration_checkpoint(self) -> None:
        """Lazily upgrade a closed v3 Journal to schema 4 (#382 D1 P1-d)."""
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) != 3:
                raise EventJournalIntegrityError("v4 migration requires v3 history")
            verified = self._verify_records(records)
            if (
                verified.open_events
                or verified.open_recoveries
                or any(t.terminal_lifecycle is None for t in verified.transactions)
                or any(not item.completed for item in verified.startup_reconciliations)
                or verified.open_gate_clear is not None
            ):
                raise EventJournalIntegrityError(
                    "v4 migration requires a closed v3 journal"
                )
            prior = records[-1]
            v3_migration_anchor_hash, _ = self._migration_anchors(records, 3)
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=prior.record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                schema_version=4,
                migration_previous_v3_hash=prior.record_hash,
                v3_migration_anchor_hash=v3_migration_anchor_hash,
                v4_migration_anchor_hash=prior.record_hash,
                wal_generation_id=verified.wal_generation_id,
                wal_record_id=verified.wal_record_id,
                wal_record_hash=verified.wal_record_hash,
                journal_lineage_id=verified.journal_lineage_id,
                external_reconciliation_required=verified.external_reconciliation_required,
            )
            self._verify_records((*records, checkpoint))
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    migrate_v4 = append_v4_migration_checkpoint

    # --- compaction and archive (#381 D1/D2, IP-4a) ---------------------------

    def compaction_due(self, *, retention_days: int, now: datetime | None = None) -> bool:
        """Cheap check: is the oldest rotated segment past the retention window?"""
        with self._lock:
            self._require_authority()
            rotated = self._rotated_paths_unlocked()
            if not rotated:
                return False
            segment, _ = self._read_verified_segment(rotated[0][1])
            if not segment:
                return False
            cutoff = (now or self._clock()) - timedelta(days=retention_days)
            return segment[-1].timestamp < cutoff

    def compact(
        self,
        *,
        retention_days: int,
        now: datetime | None = None,
        carried_tail: tuple[str, str] | None = None,
        guard: Callable[[int], bool] | None = None,
    ) -> EventJournalCompactionOutcome:
        """Archive whole rotated segments older than the retention window.

        At a clean boundary only. Steps (#381 D1 §3, D2 P1-2):
        1. choose the oldest rotated segments whose last record is older than
           ``now - retention_days`` (never the active segment);
        2. write their bytes, gzip-compressed, to a temporary file, fsync,
           rename, fsync the directory, chmod 0400; decompress and re-verify
           record hashes, chain, count and root, or abort;
        3. append COMPACTION declaring the archived range and the minimal
           carried state at the new oldest checkpoint;
        4. unlink the archived segments oldest first, fsyncing each.

        ``carried_tail`` is the (record_id, record_hash) of the Semantic
        checkpoint tail (C8): carried if it is archived now or was carried
        before. ``guard`` receives the processing high-water of the range about
        to be archived and returns True to defer (e.g. Semantic receipts that
        still need that evidence, #381 D1 P0-2); it runs under the Journal
        lock before anything is written.
        """
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, int)
            or not 0 <= retention_days <= 36_500
        ):
            raise ValueError("retention_days is invalid")
        with self._lock:
            self._require_authority()
            # Never re-archive records whose previous compaction is still
            # waiting for its unlinks (IP-4a review P2-3).
            self._roll_forward_compaction_unlocked()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) < 4:
                return EventJournalCompactionOutcome(False, "journal_schema_below_4")
            pending = self._latest_compaction(records)
            if (
                pending is not None
                and pending.boundary_checkpoint_hash != records[0].record_hash
            ):
                return EventJournalCompactionOutcome(False, "previous_compaction_pending")
            verified = self._verify_records(records)
            if (
                verified.open_events
                or verified.open_recoveries
                or any(t.terminal_lifecycle is None for t in verified.transactions)
                or any(not item.completed for item in verified.startup_reconciliations)
                or verified.external_reconciliation_required
                or verified.open_gate_clear is not None
                or verified.open_wal_rollover is not None
                or verified.journal_lineage_id is None
            ):
                return EventJournalCompactionOutcome(False, "boundary_not_clean")
            as_of = now or self._clock()
            cutoff = as_of - timedelta(days=retention_days)
            artifacts = self._journal_artifacts_unlocked()
            archived_paths: list[Path] = []
            archived_count = 0
            for segment_index, segment_path in artifacts[:-1]:
                if segment_index is None:
                    break
                segment, _ = self._read_verified_segment(segment_path)
                if segment[-1].timestamp >= cutoff:
                    break
                archived_paths.append(segment_path)
                archived_count += len(segment)
            if not archived_paths:
                return EventJournalCompactionOutcome(False, "nothing_due")
            archived = records[:archived_count]
            boundary_record = records[archived_count]
            candidate = records[: archived_count + 1]
            if any(
                later.timestamp < earlier.timestamp
                for earlier, later in zip(candidate, candidate[1:])
            ):
                # The window is judged from record timestamps only; refuse when
                # the archived range and its boundary are not monotonic (#381
                # §8). Only this range matters: a later rollback does not block
                # compaction forever (IP-4a review P1-4).
                return EventJournalCompactionOutcome(False, "clock_not_monotonic")
            if (
                boundary_record.lifecycle is not EventLifecycle.CHECKPOINT
                or boundary_record.processing_sequence is None
                or archived[0].processing_sequence is None
            ):
                raise EventJournalIntegrityError("compaction boundary is invalid")
            if guard is not None and guard(boundary_record.processing_sequence):
                return EventJournalCompactionOutcome(False, "deferred_semantic")
            previous = next(
                (
                    record
                    for record in reversed(records)
                    if record.lifecycle is EventLifecycle.COMPACTION
                ),
                None,
            )
            injected = next(
                (
                    record
                    for record in records
                    if record.lifecycle is EventLifecycle.COMPACTION
                    and record.compaction is not None
                    and record.compaction.boundary_checkpoint_hash
                    == records[0].record_hash
                ),
                None,
            )
            baseline, rollback, cleared = self._declared_state_at(
                records, archived_count, injected
            )
            tail_record: EventJournalRecord | None = None
            if carried_tail is not None:
                prior = (
                    ()
                    if injected is None or injected.compaction is None
                    else injected.compaction.carried_records
                )
                tail_record = next(
                    (
                        record
                        for record in (*archived, *prior)
                        if (record.record_id, record.record_hash) == carried_tail
                    ),
                    None,
                )
            compaction_id = str(uuid4())
            archive_name = (
                f"{archived[0].processing_sequence}-"
                f"{boundary_record.processing_sequence}-{compaction_id}.jsonl.gz"
            )
            raw = b"".join(path.read_bytes() for path in archived_paths)
            archive_path, compressed = self._write_archive_unlocked(archive_name, raw)
            try:
                self._verify_archive_bytes(compressed, archived)
            except Exception:
                self._unlink_quietly(archive_path)
                raise
            declared = EventJournalCompaction(
                compaction_id=compaction_id,
                boundary_checkpoint_hash=boundary_record.record_hash,
                archived_first_record_hash=archived[0].record_hash,
                archived_last_record_hash=archived[-1].record_hash,
                archived_record_count=len(archived),
                archived_processing_floor=archived[0].processing_sequence,
                archived_processing_high_water=boundary_record.processing_sequence,
                archived_first_timestamp=archived[0].timestamp,
                archived_last_timestamp=archived[-1].timestamp,
                archived_root=self._archived_root(archived),
                archive_file=archive_name,
                archive_sha256=hashlib.sha256(compressed).hexdigest(),
                archive_raw_sha256=hashlib.sha256(raw).hexdigest(),
                archive_bytes=len(compressed),
                retention_days=retention_days,
                as_of=as_of,
                previous_compaction_hash=(
                    None if previous is None else previous.record_hash
                ),
                baseline_record=baseline,
                true_rollback_completed_record=rollback,
                true_rollback_cleared_record=cleared,
                carried_tail_record=tail_record,
            )
            try:
                record = self._make_record(
                    EventLifecycle.COMPACTION,
                    previous_hash=records[-1].record_hash,
                    event=None,
                    schema_version=self._active_schema(records),
                    compaction=declared,
                    journal_lineage_id=verified.journal_lineage_id,
                )
                self._verify_records((*records, record))
            except EventJournalAppendError:
                self._unlink_quietly(archive_path)
                raise
            except (ValidationError, ValueError, EventJournalIntegrityError):
                self._unlink_quietly(archive_path)
                raise EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                ) from None
            self._append_record_unlocked(record)
            self._unlink_archived_segments_unlocked(archived_paths)
            return EventJournalCompactionOutcome(
                True,
                None,
                len(archived_paths),
                len(archived),
                archive_name,
            )

    def _write_archive_unlocked(self, name: str, raw: bytes) -> tuple[Path, bytes]:
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        if not self.archive_dir.exists():
            self.archive_dir.mkdir(mode=0o700)
            # The directory entry itself must survive a power loss.
            self._fsync_directory(self.archive_dir.parent)
        temporary = self.archive_dir / f".compaction-{uuid4()}.tmp"
        target = self.archive_dir / name
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(compressed)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            # Guards against mistakes, not tampering (#381 D2 P2).
            os.chmod(target, 0o400)
            self._fsync_directory(self.archive_dir)
        except BaseException:
            self._unlink_quietly(temporary)
            raise
        return target, compressed

    def _verify_archive_bytes(
        self, compressed: bytes, expected: tuple[EventJournalRecord, ...]
    ) -> None:
        archived = self._parse_segment_bytes(gzip.decompress(compressed))
        if len(archived) != len(expected):
            raise EventJournalIntegrityError("archive record count differs")
        self._verify_segment_records(archived, None)
        if tuple(item.record_hash for item in archived) != tuple(
            item.record_hash for item in expected
        ):
            raise EventJournalIntegrityError("archive records differ")

    def _unlink_archived_segments_unlocked(self, paths: list[Path]) -> None:
        for segment_path in paths:
            segment_path.unlink()
            self._fsync_parent()

    def _roll_forward_compaction_unlocked(self) -> None:
        """Finish an interrupted compaction before verification (#381 D1 P1-3).

        If the latest COMPACTION's boundary is not the oldest record, the
        archived segments are still present: unlink them oldest first, but
        only after the archive is present and matches its declared digest.
        """
        records = self._read_records_unlocked()
        latest = next(
            (
                record
                for record in reversed(records)
                if record.lifecycle is EventLifecycle.COMPACTION
            ),
            None,
        )
        if latest is None or latest.compaction is None:
            return
        declared = latest.compaction
        if records[0].record_hash == declared.boundary_checkpoint_hash:
            return
        if self._archive_problem(declared) is not None:
            # Keep every segment: nothing is lost, and the verifier replays the
            # archived range normally until the archive is restored.
            return
        boundary_hash = declared.boundary_checkpoint_hash
        doomed: list[Path] = []
        for segment_index, segment_path in self._journal_artifacts_unlocked():
            segment, _ = self._read_verified_segment(segment_path)
            if segment[0].record_hash == boundary_hash:
                break
            if segment_index is None:
                return
            doomed.append(segment_path)
        else:
            return
        raw = b"".join(segment_path.read_bytes() for segment_path in doomed)
        if hashlib.sha256(raw).hexdigest() != declared.archive_raw_sha256:
            # The present segments are not exactly what was archived: keep them.
            alert = EventJournalArchiveAlert(declared.archive_file, "segments_differ")
            if alert not in self.archive_alerts:
                self.archive_alerts = (*self.archive_alerts, alert)
                LOGGER.warning(
                    "EventJournal archive alert: %s %s",
                    alert.archive_file,
                    alert.reason,
                )
            return
        self._unlink_archived_segments_unlocked(doomed)

    def _archive_problem(self, declared: EventJournalCompaction) -> str | None:
        # Only archives named by retained COMPACTIONs are checked; those named
        # by already archived compactions are not (documented limit).
        archive = self.archive_dir / declared.archive_file
        try:
            data = archive.read_bytes()
        except FileNotFoundError:
            return "missing"
        except OSError:
            return "unreadable"
        if len(data) != declared.archive_bytes:
            return "size_mismatch"
        if hashlib.sha256(data).hexdigest() != declared.archive_sha256:
            return "digest_mismatch"
        return None

    def _check_archives_unlocked(
        self, records: tuple[EventJournalRecord, ...]
    ) -> tuple[EventJournalArchiveAlert, ...]:
        """Light check of every retained COMPACTION's archive (no fail-stop)."""
        alerts = []
        for record in records:
            if record.lifecycle is not EventLifecycle.COMPACTION:
                continue
            assert record.compaction is not None
            problem = self._archive_problem(record.compaction)
            if problem is not None:
                alerts.append(
                    EventJournalArchiveAlert(record.compaction.archive_file, problem)
                )
        return tuple(alerts)

    def _quarantine_unreferenced_archives_unlocked(
        self, records: tuple[EventJournalRecord, ...]
    ) -> None:
        """Move aside archives a crash left before their COMPACTION existed.

        Such an archive covers a range beyond the latest compaction's
        high-water. Older archives (referenced by already archived
        compactions) are kept.
        """
        try:
            entries = list(self.archive_dir.iterdir())
        except FileNotFoundError:
            return
        latest = self._latest_compaction(records) if records else None
        high_water = -1 if latest is None else latest.archived_processing_high_water
        referenced = {
            record.compaction.archive_file
            for record in records
            if record.lifecycle is EventLifecycle.COMPACTION
            and record.compaction is not None
        }
        for entry in entries:
            if entry.name.startswith(".compaction-") and entry.name.endswith(".tmp"):
                self._unlink_quietly(entry)
                continue
            match = _ARCHIVE_NAME_PATTERN.fullmatch(entry.name)
            if (
                match is None
                or entry.name in referenced
                or int(match.group(2)) <= high_water
            ):
                continue
            try:
                os.replace(entry, entry.with_name(f".{entry.name}.unreferenced"))
            except OSError:
                continue

    def _fsync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _unlink_quietly(target: Path) -> None:
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        try:
            target.unlink()
        except OSError:
            pass

    @contextmanager
    def wal_rollover_boundary(self) -> Iterator[EventJournalWalRolloverBoundary]:
        """Hold the Journal lock for a whole StateWAL rollover (#382 D1 P2).

        Lock order is Journal -> WAL: the caller performs every WAL step of the
        rollover inside this block. The yielded boundary says whether the
        Journal is at a clean point. ACCEPTED-only events may be queued, since
        they have not touched the WAL (#382 D1 P1-4).
        """
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                raise EventJournalIntegrityError("WAL rollover requires history")
            verified = self._verify_records(records)
            reason: str | None = None
            if self._active_schema(records) < 4:
                reason = "journal_schema_below_4"
            elif verified.open_wal_rollover is not None:
                reason = "wal_rollover_open"
            elif any(
                item.lifecycle is not EventLifecycle.ACCEPTED
                for item in verified.open_events
            ):
                reason = "event_in_progress"
            elif (
                verified.open_recoveries
                or any(t.terminal_lifecycle is None for t in verified.transactions)
                or any(not item.completed for item in verified.startup_reconciliations)
                or verified.open_gate_clear is not None
            ):
                reason = "maintenance_open"
            elif verified.external_reconciliation_required:
                reason = "external_reconciliation_required"
            elif (
                verified.wal_generation_id is None
                or verified.wal_snapshot_sequence != verified.snapshot_sequence
                or verified.wal_snapshot_hash != verified.snapshot_hash
            ):
                reason = "wal_anchor_not_current"
            yield EventJournalWalRolloverBoundary(
                clean=reason is None,
                reason=reason,
                processing_high_water=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                wal_generation_id=verified.wal_generation_id,
                wal_record_id=verified.wal_record_id,
                wal_record_hash=verified.wal_record_hash,
            )

    def append_wal_rollover_prepared(
        self,
        *,
        rollover_id: str,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        wal_predecessor_generation_id: str,
        wal_predecessor_record_hash: str,
    ) -> None:
        """Announce a written, not yet published StateWAL generation."""
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) < 4:
                raise EventJournalIntegrityError("WAL rollover requires schema 4")
            verified = self._verify_records(records)
            self._append_rollover_unlocked(
                records,
                EventLifecycle.WAL_GENERATION_ROLLED_PREPARED,
                rollover_id=rollover_id,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                wal_predecessor_generation_id=wal_predecessor_generation_id,
                wal_predecessor_record_hash=wal_predecessor_record_hash,
                journal_lineage_id=verified.journal_lineage_id,
            )

    def append_wal_rollover_completed(self, rollover_id: str) -> None:
        """Record that the prepared generation's manifest is published."""
        self._resolve_rollover(
            EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED, rollover_id
        )

    def append_wal_rollover_aborted(self, rollover_id: str) -> None:
        """Record that the prepared generation was never published."""
        self._resolve_rollover(EventLifecycle.WAL_GENERATION_ROLLED_ABORTED, rollover_id)

    def _resolve_rollover(self, lifecycle: EventLifecycle, rollover_id: str) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                raise EventJournalIntegrityError("WAL rollover requires history")
            verified = self._verify_records(records)
            prepared = verified.open_wal_rollover
            if prepared is None or prepared.rollover_id != rollover_id:
                raise EventJournalIntegrityError("WAL rollover is not open")
            self._append_rollover_unlocked(
                records,
                lifecycle,
                rollover_id=prepared.rollover_id,
                snapshot_sequence=prepared.snapshot_sequence,
                snapshot_hash=prepared.snapshot_hash,
                wal_generation_id=prepared.wal_generation_id,
                wal_record_id=prepared.wal_record_id,
                wal_record_hash=prepared.wal_record_hash,
                wal_predecessor_generation_id=prepared.wal_predecessor_generation_id,
                wal_predecessor_record_hash=prepared.wal_predecessor_record_hash,
                journal_lineage_id=verified.journal_lineage_id,
            )
            self._maybe_rotate_unlocked()

    def _append_rollover_unlocked(
        self,
        records: tuple[EventJournalRecord, ...],
        lifecycle: EventLifecycle,
        **fields: object,
    ) -> None:
        try:
            record = self._make_record(
                lifecycle,
                previous_hash=records[-1].record_hash,
                event=None,
                schema_version=self._active_schema(records),
                **fields,
            )
            self._verify_records((*records, record))
        except (ValidationError, ValueError, EventJournalIntegrityError):
            raise EventJournalAppendError(
                EventJournalAppendStage.VALIDATE, published=False
            ) from None
        self._append_record_unlocked(record)

    def _append_transaction(
        self, lifecycle: EventLifecycle, event: AgentEvent, **fields: object
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) < 3:
                raise EventJournalIntegrityError(
                    "transaction evidence requires schema 3"
                )
            validation_failure: EventJournalAppendError | None = None
            try:
                record = self._make_record(
                    lifecycle,
                    previous_hash=records[-1].record_hash,
                    event=event,
                    schema_version=self._active_schema(records),
                    processing_sequence=event.processing_sequence,
                    **fields,
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                validation_failure = EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                )
            if validation_failure is not None:
                raise validation_failure
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def append_transaction_prepared(
        self,
        event: AgentEvent,
        transaction_id: str,
        kind: TransactionKind,
        required_participants: tuple[ParticipantRequirement, ...],
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_PREPARED,
            event,
            transaction_id=transaction_id,
            transaction_kind=kind,
            required_participants=required_participants,
        )

    def append_participant_finalized(
        self,
        event: AgentEvent,
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
        outcome: ParticipantOutcome,
    ) -> None:
        self._append_transaction(
            EventLifecycle.PARTICIPANT_FINALIZED,
            event,
            transaction_id=transaction_id,
            participant_id=participant_id,
            operation_digest=operation_digest,
            participant_outcome=outcome,
        )

    def append_transaction_completed(
        self, event: AgentEvent, transaction_id: str
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_COMPLETED, event, transaction_id=transaction_id
        )

    def append_transaction_reconciliation_required(
        self,
        event: AgentEvent,
        transaction_id: str,
        reason: ReconciliationReason,
        unresolved_participants: tuple[str, ...],
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
            event,
            transaction_id=transaction_id,
            reconciliation_reason=reason,
            unresolved_participants=unresolved_participants,
        )

    def append_transaction_reconciled(
        self, event: AgentEvent, transaction_id: str
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_RECONCILED, event, transaction_id=transaction_id
        )

    def append_participant_aborted(
        self,
        event: AgentEvent,
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
        outcome: AbortOutcome,
    ) -> None:
        self._append_transaction(
            EventLifecycle.PARTICIPANT_ABORTED,
            event,
            transaction_id=transaction_id,
            participant_id=participant_id,
            operation_digest=operation_digest,
            abort_outcome=outcome,
        )

    def append_transaction_abort_required(
        self,
        event: AgentEvent,
        transaction_id: str,
        reason: ReconciliationReason,
        unresolved_participants: tuple[str, ...],
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_ABORT_REQUIRED,
            event,
            transaction_id=transaction_id,
            abort_reason=reason,
            unresolved_participants=unresolved_participants,
        )

    def append_transaction_aborted(
        self, event: AgentEvent, transaction_id: str
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_ABORTED, event, transaction_id=transaction_id
        )

    def _append_startup_reconciliation(
        self, lifecycle: EventLifecycle, **fields: object
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) < 3:
                raise EventJournalIntegrityError(
                    "startup reconciliation evidence requires schema 3"
                )
            validation_failure: EventJournalAppendError | None = None
            try:
                if (
                    lifecycle is EventLifecycle.STARTUP_RECONCILIATION_PREPARED
                    and not any(
                        record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
                        and record.recovery_id == fields.get("recovery_id")
                        for record in records
                    )
                ):
                    raise EventJournalIntegrityError(
                        "startup reconciliation recovery evidence is unknown"
                    )
                record = self._make_record(
                    lifecycle,
                    previous_hash=records[-1].record_hash,
                    schema_version=self._active_schema(records),
                    event=None,
                    **fields,
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                validation_failure = EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                )
            if validation_failure is not None:
                raise validation_failure
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def append_startup_reconciliation_prepared(
        self,
        reconciliation_id: str,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        recovery_processing_high_water: int,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        journal_lineage_id: str,
        required_participants: tuple[ParticipantRequirement, ...],
    ) -> None:
        self._append_startup_reconciliation(
            EventLifecycle.STARTUP_RECONCILIATION_PREPARED,
            reconciliation_id=reconciliation_id,
            recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            recovery_processing_high_water=recovery_processing_high_water,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
            journal_lineage_id=journal_lineage_id,
            required_participants=required_participants,
        )

    def append_startup_participant_reconciled(
        self,
        reconciliation_id: str,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        recovery_processing_high_water: int,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        journal_lineage_id: str,
        participant_id: str,
        operation_digest: str,
        outcome: StartupParticipantOutcome,
    ) -> None:
        self._append_startup_reconciliation(
            EventLifecycle.STARTUP_PARTICIPANT_RECONCILED,
            reconciliation_id=reconciliation_id,
            recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            recovery_processing_high_water=recovery_processing_high_water,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
            journal_lineage_id=journal_lineage_id,
            participant_id=participant_id,
            operation_digest=operation_digest,
            startup_participant_outcome=outcome,
        )

    def append_startup_reconciliation_completed(
        self,
        reconciliation_id: str,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        recovery_processing_high_water: int,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        journal_lineage_id: str,
    ) -> None:
        self._append_startup_reconciliation(
            EventLifecycle.STARTUP_RECONCILIATION_COMPLETED,
            reconciliation_id=reconciliation_id,
            recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            recovery_processing_high_water=recovery_processing_high_water,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
            journal_lineage_id=journal_lineage_id,
        )

    def _append_external_gate(
        self, lifecycle: EventLifecycle, **fields: object
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) < 3:
                raise EventJournalIntegrityError("external gate requires schema 3")
            try:
                record = self._make_record(
                    lifecycle, previous_hash=records[-1].record_hash,
                    schema_version=self._active_schema(records),
                    event=None, **fields
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                raise EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                ) from None
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def append_participant_baseline(
        self, baseline_id: str, snapshot_sequence: int, snapshot_hash: str,
        processing_high_water: int, wal_generation_id: str, wal_record_id: str,
        wal_record_hash: str, journal_lineage_id: str,
        participant_registry: tuple[ParticipantBaseline, ...],
        *, adoption_epoch: int | None = None,
    ) -> None:
        inspection = self.inspect()
        identifiers = tuple(item.participant_id for item in participant_registry)
        if (
            not participant_registry
            or identifiers != tuple(sorted(identifiers))
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("participant baseline registry is invalid")
        latest = inspection.baselines[-1] if inspection.baselines else None
        if latest is None:
            if adoption_epoch not in {None, 0}:
                raise ValueError("initial participant baseline epoch is invalid")
            record_adoption_epoch = adoption_epoch
        else:
            previous = {
                item.participant_id: item.domain
                for item in latest.participant_registry
            }
            current = {
                item.participant_id: item.domain
                for item in participant_registry
            }
            if (
                not set(current).issuperset(previous)
                or set(current) == set(previous)
                or any(
                    current[item] != domain
                    for item, domain in previous.items()
                )
            ):
                raise ValueError("participant baseline registry is not a strict superset")
            expected_epoch = latest.adoption_epoch + 1
            if expected_epoch > MAX_PERSISTED_REVISION:
                raise ValueError("participant baseline adoption epoch capacity is exhausted")
            if adoption_epoch is None:
                record_adoption_epoch = expected_epoch
            elif adoption_epoch != expected_epoch:
                raise ValueError("participant baseline adoption epoch is not monotonic")
            else:
                record_adoption_epoch = adoption_epoch
            if any(record.baseline_id == baseline_id for record in inspection.records):
                raise ValueError("participant baseline identifier is duplicated")
        if (
            inspection.open_events
            or inspection.open_transactions
            or inspection.open_recoveries
            or inspection.open_startup_reconciliations
            or inspection.open_gate_clear is not None
            or inspection.external_reconciliation_required
            or snapshot_sequence != inspection.snapshot_sequence
            or snapshot_hash != inspection.snapshot_hash
            or processing_high_water != inspection.processing_high_water
            or wal_generation_id != inspection.wal_generation_id
            or wal_record_id != inspection.wal_record_id
            or wal_record_hash != inspection.wal_record_hash
            or journal_lineage_id != inspection.journal_lineage_id
        ):
            raise EventJournalIntegrityError(
                "participant baseline requires clean current authority"
            )
        self._append_external_gate(
            EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED,
            baseline_id=baseline_id, snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            recovery_processing_high_water=processing_high_water,
            wal_generation_id=wal_generation_id, wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash, journal_lineage_id=journal_lineage_id,
            participant_registry=participant_registry,
            adoption_epoch=record_adoption_epoch,
        )

    def append_clear_prepared(
        self, baseline_id: str, reconciliation_id: str, recovery_id: str,
        snapshot_sequence: int, snapshot_hash: str, processing_high_water: int,
        wal_generation_id: str, wal_record_id: str, wal_record_hash: str,
        journal_lineage_id: str,
    ) -> None:
        self._append_external_gate(
            EventLifecycle.CLEAR_PREPARED, baseline_id=baseline_id,
            reconciliation_id=reconciliation_id, recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence, snapshot_hash=snapshot_hash,
            recovery_processing_high_water=processing_high_water,
            wal_generation_id=wal_generation_id, wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash, journal_lineage_id=journal_lineage_id,
        )

    def append_cleared(
        self, baseline_id: str, reconciliation_id: str, recovery_id: str,
        snapshot_sequence: int, snapshot_hash: str, processing_high_water: int,
        wal_generation_id: str, wal_record_id: str, wal_record_hash: str,
        journal_lineage_id: str,
    ) -> None:
        self._append_external_gate(
            EventLifecycle.CLEARED, baseline_id=baseline_id,
            reconciliation_id=reconciliation_id, recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence, snapshot_hash=snapshot_hash,
            recovery_processing_high_water=processing_high_water,
            wal_generation_id=wal_generation_id, wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash, journal_lineage_id=journal_lineage_id,
        )

    append_external_reconciliation_clear_prepared = append_clear_prepared
    append_external_reconciliation_cleared = append_cleared

    def append_v2_bootstrap_checkpoint(
        self,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        processing_high_water: int | None = None,
    ) -> None:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        if processing_high_water is None:
            processing_high_water = snapshot_sequence
        if (
            type(processing_high_water) is not int
            or processing_high_water < 0
            or processing_high_water > MAX_PERSISTED_EVENT_SEQUENCE
        ):
            raise ValueError("processing high-water is outside its bound")
        if processing_high_water < snapshot_sequence:
            raise ValueError("processing high-water precedes snapshot")
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if records:
                raise EventJournalIntegrityError(
                    "v2 bootstrap requires an empty journal"
                )
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=None,
                processing_sequence=processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
                schema_version=2,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=str(uuid4()),
                external_reconciliation_required=False,
            )
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    def append_v2_current_checkpoint(
        self,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
    ) -> None:
        """Bind reconciled current state to exact WAL evidence."""

        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or records[-1].schema_version < 2:
                raise EventJournalIntegrityError(
                    "current checkpoint requires v2 history"
                )
            verified = self._verify_records(records)
            if (
                verified.open_events
                or verified.snapshot_sequence != snapshot_sequence
                or verified.snapshot_hash != snapshot_hash
                or verified.journal_lineage_id is None
            ):
                raise EventJournalIntegrityError("current checkpoint is inconsistent")
            schema_version = records[-1].schema_version
            v3_migration_anchor_hash, v4_migration_anchor_hash = (
                self._migration_anchors(records, schema_version)
            )
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=records[-1].record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
                schema_version=schema_version,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=verified.journal_lineage_id,
                v3_migration_anchor_hash=v3_migration_anchor_hash,
                v4_migration_anchor_hash=v4_migration_anchor_hash,
                external_reconciliation_required=(
                    verified.external_reconciliation_required
                ),
            )
            self._verify_records((*records, checkpoint))
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    def append_accepted(self, event: AgentEvent) -> None:
        self._append_event(EventLifecycle.ACCEPTED, event)

    def append_started(self, event: AgentEvent) -> None:
        self._append_event(
            EventLifecycle.STARTED,
            event,
            processing_sequence=event.processing_sequence,
        )

    def append_prepared(
        self,
        event: AgentEvent,
        state_hash_before: str,
        state_hash_after: str,
        wal_generation_id: str | None = None,
    ) -> None:
        self._append_event(
            EventLifecycle.PREPARED,
            event,
            processing_sequence=event.processing_sequence,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            wal_generation_id=wal_generation_id,
        )

    def append_completed(
        self,
        event: AgentEvent,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str | None = None,
        wal_record_id: str | None = None,
        wal_record_hash: str | None = None,
    ) -> None:
        self._append_event(
            EventLifecycle.COMPLETED,
            event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
        )

    def append_recovery_prepared(
        self,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        category: EventRecoveryCategory,
        processing_high_water: int | None = None,
    ) -> None:
        self._append_v2_recovery(
            EventLifecycle.RECOVERY_PREPARED,
            recovery_id,
            snapshot_sequence,
            snapshot_hash,
            wal_generation_id,
            category,
            None,
            processing_high_water,
        )

    def append_recovery_completed(
        self,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        category: EventRecoveryCategory,
        external_reconciliation_required: bool,
        processing_high_water: int | None = None,
        *,
        wal_record_id: str,
        wal_record_hash: str,
    ) -> None:
        self._append_v2_recovery(
            EventLifecycle.RECOVERY_COMPLETED,
            recovery_id,
            snapshot_sequence,
            snapshot_hash,
            wal_generation_id,
            category,
            external_reconciliation_required,
            processing_high_water,
            wal_record_id,
            wal_record_hash,
        )

    def append_failed(
        self,
        event: AgentEvent,
        snapshot_sequence: int,
        snapshot_hash: str,
    ) -> None:
        self._append_event(
            EventLifecycle.FAILED,
            event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            failure_category=EventFailureCategory.HANDLER_FAILURE,
        )

    def verify_and_reconcile(
        self, snapshot_sequence: int, snapshot_hash: str
    ) -> EventJournalRecovery:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                checkpoint = self._make_record(
                    EventLifecycle.CHECKPOINT,
                    previous_hash=None,
                    processing_sequence=snapshot_sequence,
                    snapshot_sequence=snapshot_sequence,
                    snapshot_hash=snapshot_hash,
                    schema_version=1,
                )
                self._append_record_unlocked(checkpoint)
                return EventJournalRecovery(
                    processing_high_water=snapshot_sequence,
                    snapshot_sequence=snapshot_sequence,
                    snapshot_hash=snapshot_hash,
                )

            verified = self._verify_records(records)
            self._plan_reconciliation(verified, snapshot_sequence, snapshot_hash)
            processing = tuple(
                event
                for event in verified.open_events
                if event.lifecycle in {EventLifecycle.STARTED, EventLifecycle.PREPARED}
            )
            if len(processing) > 1:
                raise EventJournalIntegrityError(
                    "EventJournal has multiple interrupted handlers"
                )

            canonical_matches = (
                snapshot_sequence == verified.snapshot_sequence
                and snapshot_hash == verified.snapshot_hash
            )
            if processing:
                interrupted = processing[0]
                category = EventFailureCategory.UNCOMMITTED_AFTER_CRASH
                if interrupted.lifecycle is EventLifecycle.PREPARED:
                    if (
                        snapshot_sequence == interrupted.processing_sequence
                        and snapshot_hash == interrupted.state_hash_after
                    ):
                        category = EventFailureCategory.COMMITTED_BEFORE_CRASH
                    elif not (
                        canonical_matches
                        and snapshot_hash == interrupted.state_hash_before
                    ):
                        raise EventJournalIntegrityError(
                            "Prepared transition does not match canonical snapshot"
                        )
                elif not canonical_matches:
                    raise EventJournalIntegrityError(
                        "Started event does not match canonical snapshot"
                    )
                self._append_recovery_unlocked(
                    interrupted,
                    snapshot_sequence,
                    snapshot_hash,
                    category,
                )
                records = self._read_records_unlocked()
                verified = self._verify_records(records)
            elif not canonical_matches:
                raise EventJournalIntegrityError(
                    "Journal and canonical snapshot are inconsistent"
                )

            for accepted in tuple(
                event
                for event in verified.open_events
                if event.lifecycle is EventLifecycle.ACCEPTED
            ):
                self._append_recovery_unlocked(
                    accepted,
                    snapshot_sequence,
                    snapshot_hash,
                    EventFailureCategory.ACCEPTED_NOT_STARTED,
                )
                verified = self._verify_records(self._read_records_unlocked())

            return EventJournalRecovery(
                processing_high_water=verified.processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
            )

    def _append_event(
        self,
        lifecycle: EventLifecycle,
        event: AgentEvent,
        **fields: Any,
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                raise EventJournalIntegrityError(
                    "EventJournal requires startup reconciliation"
                )
            validation_failure: EventJournalAppendError | None = None
            try:
                verified = self._verify_records(records)
                if lifecycle is EventLifecycle.ACCEPTED and (
                    self._sequence_capacity_exhausted(verified)
                ):
                    raise EventJournalAppendError(
                        EventJournalAppendStage.VALIDATE, published=False
                    )
                if event.processing_sequence is not None and (
                    type(event.processing_sequence) is not int
                    or event.processing_sequence < 0
                    or event.processing_sequence > MAX_PERSISTED_EVENT_SEQUENCE
                ):
                    raise ValueError("event processing sequence is outside its bound")
                fields.setdefault("schema_version", self._active_schema(records))
                record = self._make_record(
                    lifecycle,
                    previous_hash=records[-1].record_hash,
                    event=event,
                    **fields,
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                validation_failure = EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                )
            if validation_failure is not None:
                raise validation_failure
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def _append_v2_recovery(
        self,
        lifecycle: EventLifecycle,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        category: EventRecoveryCategory,
        external: bool | None,
        processing_high_water: int | None,
        wal_record_id: str | None = None,
        wal_record_hash: str | None = None,
    ) -> None:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        if processing_high_water is not None and (
            type(processing_high_water) is not int
            or processing_high_water < 0
            or processing_high_water > MAX_PERSISTED_EVENT_SEQUENCE
        ):
            raise ValueError("processing high-water is outside its bound")
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) < 2:
                raise EventJournalIntegrityError("v2 recovery requires migration")
            verified = self._verify_records(records)
            if processing_high_water is None:
                processing_high_water = verified.processing_high_water
            if processing_high_water != verified.processing_high_water:
                raise EventJournalIntegrityError("recovery high-water changed")
            current = (verified.snapshot_sequence, verified.snapshot_hash)
            target = (snapshot_sequence, snapshot_hash)
            if category in {
                EventRecoveryCategory.EXACT_CURRENT,
                EventRecoveryCategory.UNCOMMITTED_TAIL,
            }:
                if target != current or (
                    external is not None
                    and external is not verified.external_reconciliation_required
                ):
                    raise EventJournalIntegrityError(
                        "current recovery anchor is invalid"
                    )
            elif category is EventRecoveryCategory.TRUE_ROLLBACK:
                if snapshot_sequence >= verified.snapshot_sequence or (
                    external is not None and external is not True
                ):
                    raise EventJournalIntegrityError(
                        "rollback recovery anchor is invalid"
                    )
            if lifecycle is EventLifecycle.RECOVERY_COMPLETED and (
                wal_record_id is None or wal_record_hash is None
            ):
                raise EventJournalIntegrityError("recovery WAL identity is incomplete")
            kwargs: dict[str, Any] = {
                "schema_version": self._active_schema(records),
                "recovery_id": recovery_id,
                "snapshot_sequence": snapshot_sequence,
                "snapshot_hash": snapshot_hash,
                "wal_generation_id": wal_generation_id,
                "recovery_category": category,
                "recovery_processing_high_water": processing_high_water,
            }
            if external is not None:
                kwargs["external_reconciliation_required"] = external
            if wal_record_id is not None:
                kwargs["wal_record_id"] = wal_record_id
            if wal_record_hash is not None:
                kwargs["wal_record_hash"] = wal_record_hash
            record = self._make_record(
                lifecycle, previous_hash=records[-1].record_hash, **kwargs
            )
            self._verify_records((*records, record))
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def _append_recovery_unlocked(
        self,
        event: _EventState,
        snapshot_sequence: int,
        snapshot_hash: str,
        category: EventFailureCategory,
    ) -> None:
        records = self._read_records_unlocked()
        journal_event = AgentEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            source=event.source,
            requested_at=event.requested_at,
            processing_sequence=event.processing_sequence,
        )
        record = self._make_record(
            EventLifecycle.RECOVERY_CLASSIFIED,
            previous_hash=records[-1].record_hash,
            event=journal_event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            failure_category=category,
            schema_version=self._active_schema(records),
        )
        self._verify_records((*records, record))
        self._append_record_unlocked(record)
        self._maybe_rotate_unlocked()

    def _make_record(
        self,
        lifecycle: EventLifecycle,
        *,
        previous_hash: str | None,
        event: AgentEvent | None = None,
        schema_version: int | None = None,
        **fields: object,
    ) -> EventJournalRecord:
        validation_failure: EventJournalAppendError | None = None
        try:
            timestamp = self._clock()
            unsigned = EventJournalRecord.model_validate(
                {
                    "schema_version": schema_version or 1,
                    "record_id": str(uuid4()),
                    "timestamp": timestamp,
                    "lifecycle": lifecycle,
                    "event_id": event.event_id if event is not None else None,
                    "event_type": event.event_type if event is not None else None,
                    "source": event.source if event is not None else None,
                    "previous_record_hash": previous_hash,
                    "record_hash": "0" * 64,
                    **fields,
                }
            )
            record = unsigned.model_copy(
                update={"record_hash": self._record_hash(unsigned)}
            )
        except Exception:
            validation_failure = EventJournalAppendError(
                EventJournalAppendStage.VALIDATE, published=False
            )
        if validation_failure is not None:
            raise validation_failure
        return record

    @staticmethod
    def _record_hash(record: EventJournalRecord) -> str:
        excluded = (
            {
                "wal_generation_id",
                "wal_record_id",
                "wal_record_hash",
                "migration_previous_v1_hash",
                "journal_lineage_id",
                "recovery_id",
                "recovery_category",
                "recovery_processing_high_water",
                "external_reconciliation_required",
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
                "baseline_id",
                "participant_registry",
            }
            if record.schema_version == 1
            else {
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
                "baseline_id",
                "participant_registry",
            }
            if record.schema_version in {1, 2}
            else set()
        )
        if record.schema_version < 4:
            excluded.update(_V4_FIELDS)
        if record.adoption_epoch is None:
            excluded.add("adoption_epoch")
        canonical = json.dumps(
            record.model_dump(mode="json", exclude={"record_hash", *excluded}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        domain = (
            _HASH_DOMAIN_V1
            if record.schema_version == 1
            else _HASH_DOMAIN_V2
            if record.schema_version == 2
            else _HASH_DOMAIN_V3
            if record.schema_version == 3
            else _HASH_DOMAIN_V4
        )
        return hashlib.sha256(domain + canonical).hexdigest()

    @staticmethod
    def _record_bytes(record: EventJournalRecord) -> bytes:
        excluded = (
            {
                "wal_generation_id",
                "wal_record_id",
                "wal_record_hash",
                "migration_previous_v1_hash",
                "journal_lineage_id",
                "recovery_id",
                "recovery_category",
                "recovery_processing_high_water",
                "external_reconciliation_required",
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
                "baseline_id",
                "participant_registry",
            }
            if record.schema_version == 1
            else {
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
                "baseline_id",
                "participant_registry",
            }
            if record.schema_version in {1, 2}
            else set()
        )
        if record.schema_version < 4:
            excluded.update(_V4_FIELDS)
        if record.adoption_epoch is None:
            excluded.add("adoption_epoch")
        return (
            json.dumps(
                record.model_dump(mode="json", exclude=excluded),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def _append_record_unlocked(self, record: EventJournalRecord) -> None:
        stage = EventJournalAppendStage.WRITE
        published = False
        append_failure: EventJournalAppendError | None = None
        descriptor: int | None = None
        try:
            self._run_hook(record.lifecycle, stage)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("journal is not a regular file")
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab") as journal_file:
                descriptor = None
                payload = self._record_bytes(record)
                if journal_file.write(payload) != len(payload):
                    raise OSError("incomplete journal write")
                journal_file.flush()
                stage = EventJournalAppendStage.FILE_FSYNC
                self._run_hook(record.lifecycle, stage)
                os.fsync(journal_file.fileno())
                published = True
            stage = EventJournalAppendStage.PARENT_FSYNC
            self._run_hook(record.lifecycle, stage)
            self._fsync_parent()
        except Exception:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            append_failure = EventJournalAppendError(stage, published=published)
        if append_failure is not None:
            raise append_failure

    def _maybe_rotate_unlocked(self) -> None:
        failure: EventJournalAppendError | None
        try:
            size = self.path.stat().st_size
        except OSError:
            failure = EventJournalAppendError(
                EventJournalAppendStage.ROTATION, published=True
            )
        else:
            failure = None
        if failure is not None:
            raise failure
        if size <= self.max_bytes:
            return
        safe_rotation, _lifecycle_blocked, _proof_blocked = (
            self._rotation_decision_unlocked()
        )
        if not safe_rotation:
            return
        verified = self._verify_records(self._read_records_unlocked())
        self._rotate_unlocked(verified)

    def _admission_status_unlocked(self) -> EventJournalAdmissionStatus:
        status = self._segment_admission_status_unlocked()
        try:
            live_bytes = status.active_file_bytes + sum(
                path.stat().st_size for _index, path in self._rotated_paths_unlocked()
            )
        except OSError as error:
            raise EventJournalLoadError(
                "EventJournal size cannot be inspected"
            ) from error
        band = EventJournalAdmissionBand.NORMAL
        soft_limit: int | None = None
        if self.max_total_bytes is not None:
            soft_limit = self.max_total_bytes - self.soft_band_bytes
            if live_bytes > self.max_total_bytes:
                band = EventJournalAdmissionBand.HARD
            elif live_bytes > soft_limit:
                band = EventJournalAdmissionBand.SOFT
        if band is not self._last_band:
            LOGGER.warning(
                "EventJournal capacity band %s -> %s (live %d bytes, "
                "soft limit %s, hard limit %s)",
                self._last_band.value,
                band.value,
                live_bytes,
                soft_limit,
                self.max_total_bytes,
            )
            self._last_band = band
        available = status.available
        reason = status.reason
        if band is EventJournalAdmissionBand.HARD and available:
            # Hard limit: fail-stop for new admission, as before (#381 §6).
            # Its own reason, because it is the one the Human acts on (raise
            # max_total_bytes or shorten retention_days; #381 D2 P2).
            available = False
            reason = "journal_hard_limit"
        elif band is EventJournalAdmissionBand.SOFT and available:
            reason = "journal_soft_limit"
        return replace(
            status,
            available=available,
            reason=reason,
            band=band,
            live_bytes=live_bytes,
            max_total_bytes=self.max_total_bytes,
            soft_limit_bytes=soft_limit,
        )

    def _segment_admission_status_unlocked(self) -> EventJournalAdmissionStatus:
        try:
            active_bytes = self.path.stat().st_size
        except OSError as error:
            raise EventJournalLoadError(
                "EventJournal size cannot be inspected"
            ) from error
        rotated_count = len(self._rotated_paths_unlocked())
        records = self._read_records_unlocked()
        if records and self._sequence_capacity_exhausted(
            self._verify_records(records)
        ):
            return EventJournalAdmissionStatus(
                False,
                "processing_sequence_exhausted",
                active_bytes,
                self.max_bytes,
                rotated_count,
                self.retained_files,
                False,
                False,
                False,
            )
        if active_bytes <= self.max_bytes:
            return EventJournalAdmissionStatus(
                True,
                None,
                active_bytes,
                self.max_bytes,
                rotated_count,
                self.retained_files,
                False,
                False,
                False,
            )
        safe_rotation, lifecycle_blocked, proof_blocked = (
            self._rotation_decision_unlocked()
        )
        return EventJournalAdmissionStatus(
            safe_rotation,
            None if safe_rotation else "event_journal_retention_exhausted",
            active_bytes,
            self.max_bytes,
            rotated_count,
            self.retained_files,
            safe_rotation,
            lifecycle_blocked,
            proof_blocked,
        )

    @staticmethod
    def _sequence_capacity_exhausted(verified: _VerifiedJournal) -> bool:
        reserved = sum(
            state.lifecycle is EventLifecycle.ACCEPTED
            for state in verified.open_events
        )
        return (
            verified.processing_high_water + reserved
            >= MAX_PERSISTED_EVENT_SEQUENCE
        )

    def _rotation_decision_unlocked(self) -> tuple[bool, bool, bool]:
        """Use one authority-preserving policy for status and ordinary rotation."""
        records = self._read_records_unlocked()
        if not records:
            raise EventJournalLoadError("EventJournal cannot rotate without records")
        verified = self._verify_records(records)
        lifecycle_blocked = bool(
            verified.open_events
            or verified.open_recoveries
            or any(t.terminal_lifecycle is None for t in verified.transactions)
            or any(not item.completed for item in verified.startup_reconciliations)
            or verified.external_reconciliation_required
            or verified.open_gate_clear is not None
            or verified.open_wal_rollover is not None
        )
        # Retention is by time and compaction (#381 D1 P1-4), never by a
        # segment count: proof retention no longer blocks rotation.
        proof_blocked = False
        anchor_blocked = verified.records[-1].schema_version >= 2 and not any(
            record.lifecycle in _WAL_ANCHOR_LIFECYCLES
            and record.schema_version >= 2
            and record.snapshot_sequence == verified.snapshot_sequence
            and record.snapshot_hash == verified.snapshot_hash
            and record.wal_generation_id is not None
            and record.wal_record_id is not None
            and record.wal_record_hash is not None
            for record in reversed(verified.records)
        )
        return (
            not (lifecycle_blocked or proof_blocked or anchor_blocked),
            lifecycle_blocked,
            proof_blocked,
        )

    def _rotate_unlocked(self, verified: _VerifiedJournal) -> None:
        failure: EventJournalAppendError | None = None
        try:
            rotated = self._rotated_paths_unlocked()
            next_index = rotated[-1][0] + 1 if rotated else 0
            archived = self._segment_path(next_index)
            wal_generation_id: str | None = None
            wal_record_id: str | None = None
            wal_record_hash: str | None = None
            journal_lineage_id: str | None = None
            v3_migration_anchor_hash: str | None = None
            v4_migration_anchor_hash: str | None = None
            if verified.records[-1].schema_version >= 2:
                anchor = next(
                    (
                        record
                        for record in reversed(verified.records)
                        if record.lifecycle
                        in _WAL_ANCHOR_LIFECYCLES
                        and record.schema_version >= 2
                        and record.wal_generation_id is not None
                        and record.wal_record_id is not None
                        and record.wal_record_hash is not None
                    ),
                    None,
                )
                if anchor is None:
                    raise EventJournalIntegrityError("v2 Journal lacks WAL anchor")
                if (
                    anchor.snapshot_sequence != verified.snapshot_sequence
                    or anchor.snapshot_hash != verified.snapshot_hash
                ):
                    raise EventJournalIntegrityError("v2 Journal anchor is stale")
                wal_generation_id = anchor.wal_generation_id
                wal_record_id = anchor.wal_record_id
                wal_record_hash = anchor.wal_record_hash
                journal_lineage_id = verified.journal_lineage_id
                v3_migration_anchor_hash, v4_migration_anchor_hash = (
                    self._migration_anchors(
                        verified.records, verified.records[-1].schema_version
                    )
                )
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=verified.records[-1].record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                schema_version=verified.records[-1].schema_version,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=journal_lineage_id,
                v3_migration_anchor_hash=v3_migration_anchor_hash,
                v4_migration_anchor_hash=v4_migration_anchor_hash,
                external_reconciliation_required=(
                    verified.external_reconciliation_required
                    if verified.records[-1].schema_version >= 2
                    else None
                ),
            )
            temporary = self.path.with_name(f".{self.path.name}.rotation.tmp")
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as segment_file:
                segment_file.write(self._record_bytes(checkpoint))
                segment_file.flush()
                os.fsync(segment_file.fileno())
            os.replace(self.path, archived)
            self._fsync_parent()
            os.replace(temporary, self.path)
            self._fsync_parent()
        except Exception:
            failure = EventJournalAppendError(
                EventJournalAppendStage.ROTATION, published=True
            )
        if failure is not None:
            raise failure

    def _read_records_unlocked(self) -> tuple[EventJournalRecord, ...]:
        artifacts = self._journal_artifacts_unlocked()
        if not artifacts:
            return ()
        records: list[EventJournalRecord] = []
        previous_tail: str | None = None
        live_keys: set[tuple[int, int]] = set()
        for segment_index, path in artifacts:
            segment, key = self._read_verified_segment(path)
            live_keys.add(key)
            if not segment:
                raise EventJournalLoadError("EventJournal segment is empty")
            if segment[0].lifecycle is not EventLifecycle.CHECKPOINT:
                raise EventJournalIntegrityError(
                    "EventJournal segment lacks checkpoint anchor"
                )
            if (
                previous_tail is not None
                and segment[0].previous_record_hash != previous_tail
            ):
                raise EventJournalIntegrityError("EventJournal segment chain is broken")
            if (
                previous_tail is None
                and segment_index is None
                and segment[0].previous_record_hash is not None
                and not any(
                    # Every rotated segment was archived by a compaction that
                    # names this checkpoint as its boundary (#381, IP-4a).
                    record.lifecycle is EventLifecycle.COMPACTION
                    and record.compaction is not None
                    and record.compaction.boundary_checkpoint_hash
                    == segment[0].record_hash
                    and record.compaction.archived_last_record_hash
                    == segment[0].previous_record_hash
                    for record in segment
                )
            ):
                raise EventJournalIntegrityError(
                    "EventJournal rotated predecessor is missing"
                )
            if segment_index is not None and path == self.path:
                raise EventJournalIntegrityError(
                    "EventJournal segment identity is invalid"
                )
            records.extend(segment)
            previous_tail = segment[-1].record_hash
        for stale in set(self._segment_cache) - live_keys:
            del self._segment_cache[stale]
        return tuple(records)

    def invalidate_verification_cache(self) -> None:
        """Forget cached segment parses and verifier state.

        The next read and the next ``_verify_records`` verify everything.

        Used by tests and available to diagnostics/repair tooling that wants a
        full re-verification without restarting the process.
        """
        with self._lock:
            self._segment_cache.clear()
            self._verifier_state = None

    def _verify_segment_records(
        self,
        segment: tuple[EventJournalRecord, ...],
        previous: EventJournalRecord | None,
    ) -> None:
        """Check record hashes and the intra-segment chain of ``segment``."""
        for index, record in enumerate(segment):
            if self._record_hash(record) != record.record_hash:
                raise EventJournalIntegrityError("EventJournal record hash mismatch")
            predecessor = segment[index - 1] if index > 0 else previous
            if (
                predecessor is not None
                and record.previous_record_hash != predecessor.record_hash
            ):
                raise EventJournalIntegrityError("EventJournal record chain is broken")

    def _read_verified_segment(
        self, path: Path
    ) -> tuple[tuple[EventJournalRecord, ...], tuple[int, int]]:
        """Read, parse and hash-verify one segment, reusing a verified prefix.

        Segments are only ever appended to or renamed (rotation), never
        rewritten in place, so an in-process cache keyed by (dev, inode) is
        reused when the file has not shrunk and the last verified line is
        still byte-identical at its verified position; only the appended
        lines are then parsed and verified.  Any other situation re-reads the
        whole segment.  The cache is never persisted and a new EventJournal
        (every process start) verifies everything.

        Threat model: single cooperative writer under the Journal lease.  A
        same-size in-place rewrite of an earlier verified line that keeps the
        last line intact is detected only by a full verification (process
        start or ``invalidate_verification_cache``).
        """
        read_failure: EventJournalLoadError | None = None
        descriptor: int | None = None
        try:
            status = path.lstat()
            if not stat.S_ISREG(status.st_mode):
                raise OSError("journal segment is not regular")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError("journal segment is not regular")
            key = (opened.st_dev, opened.st_ino)
            cached = self._segment_cache.get(key)
            with os.fdopen(descriptor, "rb") as segment_file:
                descriptor = None
                suffix: bytes | None = None
                if cached is not None and opened.st_size >= cached.length:
                    segment_file.seek(cached.length - len(cached.last_line))
                    data = segment_file.read()
                    if data.startswith(cached.last_line):
                        suffix = data[len(cached.last_line) :]
                    else:
                        segment_file.seek(0)
                raw = segment_file.read() if suffix is None else b""
        except OSError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            read_failure = EventJournalLoadError("EventJournal segment cannot be read")
        if read_failure is not None:
            raise read_failure
        self._segment_cache.pop(key, None)
        if suffix is not None:
            assert cached is not None
            if suffix and not suffix.endswith(b"\n"):
                raise EventJournalLoadError("EventJournal contains a partial record")
            appended = self._parse_segment_bytes(suffix) if suffix else ()
            self._verify_segment_records(appended, cached.records[-1])
            records = cached.records + appended
            length = cached.length + len(suffix)
            last_line = (
                suffix.splitlines(keepends=True)[-1] if suffix else cached.last_line
            )
        else:
            if not raw or not raw.endswith(b"\n"):
                raise EventJournalLoadError("EventJournal contains a partial record")
            records = self._parse_segment_bytes(raw)
            self._verify_segment_records(records, None)
            length = len(raw)
            last_line = raw.splitlines(keepends=True)[-1]
        self._segment_cache[key] = _CachedSegment(
            length=length, last_line=last_line, records=records
        )
        return records, key

    def _parse_segment_bytes(self, raw: bytes) -> tuple[EventJournalRecord, ...]:
        records: list[EventJournalRecord] = []
        parse_failure: EventJournalLoadError | None = None
        for line in raw.splitlines():
            try:
                value = json.loads(
                    line.decode("utf-8"),
                    parse_constant=self._reject_json_constant,
                )
                if not isinstance(value, dict):
                    raise ValueError("record root is invalid")
                if "schema_version" not in value:
                    raise ValueError("record schema version is missing")
                version = value["schema_version"]
                if (
                    isinstance(version, int)
                    and not isinstance(version, bool)
                    and version not in {1, 2, 3, 4}
                ):
                    raise UnsupportedEventJournalVersion(
                        "EventJournal schema version is unsupported"
                    )
                records.append(EventJournalRecord.model_validate_json(line))
            except UnsupportedEventJournalVersion:
                raise
            except (UnicodeError, json.JSONDecodeError, ValueError, ValidationError):
                parse_failure = EventJournalLoadError(
                    "EventJournal contains an invalid record"
                )
                break
        if parse_failure is not None:
            raise parse_failure
        return tuple(records)

    @staticmethod
    def _active_schema(records: tuple[EventJournalRecord, ...]) -> int:
        return records[-1].schema_version

    def _journal_artifacts_unlocked(self) -> list[tuple[int | None, Path]]:
        scan_failure: EventJournalLoadError | None = None
        try:
            candidates = tuple(self.path.parent.glob(f"{self.path.name}.*"))
        except OSError:
            candidates = ()
            scan_failure = EventJournalLoadError(
                "EventJournal artifacts cannot be inspected"
            )
        if scan_failure is not None:
            raise scan_failure

        rotated: list[tuple[int, Path]] = []
        interrupted_failure: EventJournalLoadError | None = None
        try:
            interrupted = tuple(self.path.parent.glob(f".{self.path.name}.rotation*"))
        except OSError:
            interrupted = ()
            interrupted_failure = EventJournalLoadError(
                "EventJournal artifacts cannot be inspected"
            )
        if interrupted_failure is not None:
            raise interrupted_failure
        if interrupted:
            temporary = self.path.with_name(f".{self.path.name}.rotation.tmp")
            try:
                active_exists = self.path.lstat() is not None
            except FileNotFoundError:
                active_exists = False
            if (
                len(interrupted) == 1
                and interrupted[0] == temporary
                and not active_exists
            ):
                try:
                    status = temporary.lstat()
                    if (
                        not stat.S_ISREG(status.st_mode)
                        or status.st_uid != os.geteuid()
                        or stat.S_IMODE(status.st_mode) != 0o600
                    ):
                        raise OSError
                    os.replace(temporary, self.path)
                    self._fsync_parent()
                except OSError:
                    raise EventJournalIntegrityError(
                        "EventJournal interrupted rotation is unsafe"
                    ) from None
            else:
                raise EventJournalIntegrityError(
                    "EventJournal has an interrupted rotation artifact"
                )
        for candidate in candidates:
            suffix = candidate.name.removeprefix(f"{self.path.name}.")
            if not re.fullmatch(r"\d{8}", suffix):
                raise EventJournalIntegrityError(
                    "EventJournal has an interrupted rotation artifact"
                )
            rotated.append((int(suffix), candidate))
        rotated.sort()
        indexes = [index for index, _ in rotated]
        if indexes and indexes != list(range(indexes[0], indexes[-1] + 1)):
            raise EventJournalIntegrityError("EventJournal rotated segment is missing")

        active_failure: EventJournalLoadError | None = None
        try:
            active_exists = self.path.lstat() is not None
        except FileNotFoundError:
            active_exists = False
        except OSError:
            active_exists = False
            active_failure = EventJournalLoadError("EventJournal cannot be inspected")
        if active_failure is not None:
            raise active_failure
        if rotated and not active_exists:
            raise EventJournalIntegrityError("EventJournal active segment is missing")
        artifacts: list[tuple[int | None, Path]] = [*rotated]
        if active_exists:
            artifacts.append((None, self.path))
        return artifacts

    def _verify_records(
        self, records: tuple[EventJournalRecord, ...]
    ) -> _VerifiedJournal:
        with self._lock:
            state = self._take_verifier_state(records)
            if state is not None and state.result is not None:
                if len(records) == len(state.records):
                    result = replace(state.result, records=records)
                    self._verifier_state = state
                    return result
            result, state = self._verify_records_from(records, state)
            self._verifier_state = state
            return result

    def _take_verifier_state(
        self, records: tuple[EventJournalRecord, ...]
    ) -> _VerifierState | None:
        """Detach the cached state if ``records`` extends its verified prefix.

        The cache is cleared before any verification work, so a failure never
        leaves a partially advanced state behind.
        """
        state = getattr(self, "_verifier_state", None)
        self._verifier_state = None
        if state is None:
            return None
        prefix = len(state.records)
        if len(records) < prefix or records[:prefix] != state.records:
            return None
        if any(
            record.lifecycle is EventLifecycle.COMPACTION
            for record in records[prefix:]
        ):
            # A new COMPACTION changes which one is latest, which decides
            # the injected initial state: verify from scratch.
            return None
        return state

    @staticmethod
    def _verify_records_from(
        records: tuple[EventJournalRecord, ...], verifier: _VerifierState | None
    ) -> tuple[_VerifiedJournal, _VerifierState]:
        if verifier is not None:
            start = len(verifier.records)
        else:
            start = 0
            verifier = EventJournal._initial_verifier_state(records)
        high_water = verifier.high_water
        snapshot_sequence = verifier.snapshot_sequence
        snapshot_hash = verifier.snapshot_hash
        journal_lineage_id = verifier.journal_lineage_id
        wal_generation_id = verifier.wal_generation_id
        wal_record_id = verifier.wal_record_id
        wal_record_hash = verifier.wal_record_hash
        wal_snapshot_sequence = verifier.wal_snapshot_sequence
        wal_snapshot_hash = verifier.wal_snapshot_hash
        external_reconciliation_required = verifier.external_reconciliation_required
        v2_seen = verifier.v2_seen
        v3_seen = verifier.v3_seen
        v3_migration_anchor_hash = verifier.v3_migration_anchor_hash
        v4_seen = verifier.v4_seen
        v4_migration_anchor_hash = verifier.v4_migration_anchor_hash
        rollover_open = verifier.rollover_open
        seen_rollover_ids = verifier.seen_rollover_ids
        last_compaction_hash = verifier.last_compaction_hash
        injected_compaction = verifier.injected_compaction
        injected_compaction_passed = verifier.injected_compaction_passed
        processing_event_id = verifier.processing_event_id
        gate_clear_open = verifier.gate_clear_open
        open_events = verifier.open_events
        accepted_queue = verifier.accepted_queue
        seen_event_ids = verifier.seen_event_ids
        seen_record_ids = verifier.seen_record_ids
        recovery_open = verifier.recovery_open
        seen_recovery_ids = verifier.seen_recovery_ids
        completed_recoveries = verifier.completed_recoveries
        startup_open = verifier.startup_open
        startup_seen_ids = verifier.startup_seen_ids
        startup_seen_recovery_ids = verifier.startup_seen_recovery_ids
        startup_completed = verifier.startup_completed
        transactions = verifier.transactions
        baseline_records_seen = verifier.baseline_records_seen
        gate_records = verifier.gate_records
        view_cache = verifier.transaction_views

        for record in records[start:]:
            if record.record_id in seen_record_ids:
                raise EventJournalIntegrityError("Record identifier is duplicated")
            seen_record_ids.add(record.record_id)

        for index in range(max(start, 1), len(records)):
            record = records[index]
            if record.lifecycle in _GATE_LIFECYCLES:
                gate_records.append(record)
            if record.transaction_id is not None:
                view_cache.pop(record.transaction_id, None)
            if record.schema_version == 4:
                if not v4_seen and (
                    record.lifecycle is not EventLifecycle.CHECKPOINT
                    or record.migration_previous_v3_hash
                    != records[index - 1].record_hash
                    or records[index - 1].schema_version != 3
                    or record.v4_migration_anchor_hash
                    != record.migration_previous_v3_hash
                ):
                    raise EventJournalIntegrityError("v4 migration boundary is invalid")
                if v4_seen and record.migration_previous_v3_hash is not None:
                    raise EventJournalIntegrityError("v4 migration is duplicated")
                v4_seen = True
            elif record.schema_version == 3:
                if v4_seen:
                    raise EventJournalIntegrityError("schema downgrade is invalid")
                if not v3_seen and (
                    record.lifecycle is not EventLifecycle.CHECKPOINT
                    or record.migration_previous_v2_hash
                    != records[index - 1].record_hash
                    or records[index - 1].schema_version != 2
                    or record.v3_migration_anchor_hash
                    != record.migration_previous_v2_hash
                ):
                    raise EventJournalIntegrityError("v3 migration boundary is invalid")
                if v3_seen and record.migration_previous_v2_hash is not None:
                    raise EventJournalIntegrityError("v3 migration is duplicated")
                v3_seen = True
            elif record.schema_version == 2:
                if v3_seen:
                    raise EventJournalIntegrityError("schema downgrade is invalid")
                if not v2_seen and (
                    record.lifecycle is not EventLifecycle.CHECKPOINT
                    or record.migration_previous_v1_hash
                    != records[index - 1].record_hash
                    or records[index - 1].schema_version != 1
                ):
                    raise EventJournalIntegrityError("v2 migration boundary is invalid")
                if v2_seen and record.migration_previous_v1_hash is not None:
                    raise EventJournalIntegrityError("v2 migration is duplicated")
                v2_seen = True
            elif v2_seen or v3_seen:
                raise EventJournalIntegrityError("v1 record follows v2 migration")
            if rollover_open is not None and record.lifecycle not in {
                EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED,
                EventLifecycle.WAL_GENERATION_ROLLED_ABORTED,
            }:
                # Only the completion or abort of an open rollover may follow
                # it: no event, checkpoint, baseline or recovery (#382 P1-b).
                raise EventJournalIntegrityError("WAL rollover is open")
            if record.lifecycle in _ROLLOVER_LIFECYCLES:
                if record.schema_version < 4:
                    raise EventJournalIntegrityError("WAL rollover requires schema 4")
                if record.lifecycle is EventLifecycle.WAL_GENERATION_ROLLED_PREPARED:
                    if record.rollover_id in seen_rollover_ids:
                        raise EventJournalIntegrityError("WAL rollover is duplicated")
                    if (
                        any(
                            item.lifecycle is not EventLifecycle.ACCEPTED
                            for item in open_events.values()
                        )
                        or processing_event_id is not None
                        or recovery_open
                        or any(t["terminal"] is None for t in transactions.values())
                        or startup_open
                        or gate_clear_open
                        or external_reconciliation_required
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                        or wal_snapshot_sequence != snapshot_sequence
                        or wal_snapshot_hash != snapshot_hash
                        or wal_generation_id is None
                        or record.wal_predecessor_generation_id != wal_generation_id
                        or record.wal_predecessor_record_hash != wal_record_hash
                        or record.journal_lineage_id != journal_lineage_id
                        or any(
                            previous.wal_generation_id == record.wal_generation_id
                            for previous in records[:index]
                        )
                    ):
                        raise EventJournalIntegrityError(
                            "WAL rollover preparation is invalid"
                        )
                    assert record.rollover_id is not None
                    seen_rollover_ids.add(record.rollover_id)
                    rollover_open = record
                    continue
                if rollover_open is None or any(
                    getattr(record, name) != getattr(rollover_open, name)
                    for name in _ROLLOVER_FIELDS
                ):
                    raise EventJournalIntegrityError(
                        "WAL rollover resolution is invalid"
                    )
                if record.lifecycle is EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED:
                    # State update rule (#382 D1 P1-a): the same snapshot now
                    # has a new WAL anchor, the new generation's baseline.
                    wal_generation_id = record.wal_generation_id
                    wal_record_id = record.wal_record_id
                    wal_record_hash = record.wal_record_hash
                rollover_open = None
                continue
            if record.lifecycle is EventLifecycle.COMPACTION:
                assert record.compaction is not None
                declared = record.compaction
                if (
                    record.schema_version < 4
                    or open_events
                    or processing_event_id is not None
                    or recovery_open
                    or any(t["terminal"] is None for t in transactions.values())
                    or startup_open
                    or gate_clear_open
                    or external_reconciliation_required
                    or record.journal_lineage_id != journal_lineage_id
                    or declared.previous_compaction_hash != last_compaction_hash
                ):
                    raise EventJournalIntegrityError("compaction is invalid")
                for carried in declared.carried_records:
                    if EventJournal._record_hash(carried) != carried.record_hash:
                        raise EventJournalIntegrityError(
                            "compaction carried record is not authentic"
                        )
                boundary = next(
                    (
                        position
                        for position in range(index)
                        if records[position].record_hash
                        == declared.boundary_checkpoint_hash
                    ),
                    None,
                )
                if boundary is None:
                    # Only a compaction OLDER than the one in effect may have
                    # lost its boundary (archived by that later compaction);
                    # every one at or after it must find its boundary (IP-4a
                    # review P1-A).
                    if injected_compaction is None or injected_compaction_passed:
                        raise EventJournalIntegrityError(
                            "compaction boundary is missing"
                        )
                elif boundary == 0:
                    if (
                        injected_compaction is None
                        or injected_compaction.record_hash != record.record_hash
                    ):
                        raise EventJournalIntegrityError(
                            "compaction state was not injected"
                        )
                    injected_compaction_passed = True
                else:
                    if injected_compaction is not None and not injected_compaction_passed:
                        raise EventJournalIntegrityError(
                            "compaction order is invalid"
                        )
                    EventJournal._verify_declared_compaction(
                        records, boundary, declared, injected_compaction
                    )
                last_compaction_hash = record.record_hash
                continue
            if record.lifecycle is EventLifecycle.CHECKPOINT:
                if open_events:
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open event lifecycle"
                    )
                if recovery_open:
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open recovery lifecycle"
                    )
                if any(t["terminal"] is None for t in transactions.values()):
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open transaction lifecycle"
                    )
                if startup_open:
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open startup reconciliation"
                    )
                if (
                    record.processing_sequence != high_water
                    or record.snapshot_sequence != snapshot_sequence
                    or record.snapshot_hash != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Checkpoint continuity is invalid")
                if (
                    record.schema_version == 3
                    and record.migration_previous_v2_hash is not None
                    and (
                        records[index - 1].schema_version != 2
                        or record.migration_previous_v2_hash
                        != records[index - 1].record_hash
                    )
                ):
                    raise EventJournalIntegrityError("v3 migration anchor is invalid")
                if record.schema_version >= 3:
                    if v3_migration_anchor_hash is None:
                        v3_migration_anchor_hash = record.v3_migration_anchor_hash
                    elif record.v3_migration_anchor_hash != v3_migration_anchor_hash:
                        raise EventJournalIntegrityError(
                            "v3 migration authority changed"
                        )
                if record.schema_version >= 4:
                    if v4_migration_anchor_hash is None:
                        v4_migration_anchor_hash = record.v4_migration_anchor_hash
                    elif record.v4_migration_anchor_hash != v4_migration_anchor_hash:
                        raise EventJournalIntegrityError(
                            "v4 migration authority changed"
                        )
                if (
                    record.schema_version >= 2
                    and record.migration_previous_v1_hash is None
                    and record.snapshot_sequence == wal_snapshot_sequence
                    and record.snapshot_hash == wal_snapshot_hash
                    and (
                        record.wal_generation_id != wal_generation_id
                        or record.wal_record_id != wal_record_id
                        or record.wal_record_hash != wal_record_hash
                        or (
                            record.external_reconciliation_required
                            is not external_reconciliation_required
                            and not (
                                record.external_reconciliation_required is False
                                and max(
                                    (
                                        position
                                        for position, previous in enumerate(
                                            records[:index]
                                        )
                                        if previous.lifecycle
                                        is EventLifecycle.CLEARED
                                    ),
                                    default=-1,
                                )
                                > max(
                                    (
                                        position
                                        for position, previous in enumerate(
                                            records[:index]
                                        )
                                        if previous.lifecycle
                                        is EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED
                                    ),
                                    default=-1,
                                )
                            )
                        )
                    )
                ):
                    raise EventJournalIntegrityError("v2 checkpoint anchor changed")
                if (
                    record.schema_version >= 2
                    and record.migration_previous_v1_hash is not None
                ):
                    if (
                        record.migration_previous_v1_hash
                        != records[index - 1].record_hash
                        or records[index - 1].schema_version != 1
                        or record.wal_generation_id is None
                        or record.wal_record_id is None
                        or record.wal_record_hash is None
                    ):
                        raise EventJournalIntegrityError(
                            "v2 migration anchor is invalid"
                        )
                if record.schema_version >= 2:
                    if journal_lineage_id is None:
                        journal_lineage_id = record.journal_lineage_id
                    elif record.journal_lineage_id != journal_lineage_id:
                        raise EventJournalIntegrityError("v2 Journal lineage changed")
                    wal_generation_id = record.wal_generation_id
                    wal_record_id = record.wal_record_id
                    wal_record_hash = record.wal_record_hash
                    wal_snapshot_sequence = record.snapshot_sequence
                    wal_snapshot_hash = record.snapshot_hash
                continue
            if record.lifecycle in {
                EventLifecycle.RECOVERY_PREPARED,
                EventLifecycle.RECOVERY_COMPLETED,
            }:
                if record.schema_version < 2 or record.recovery_id is None:
                    raise EventJournalIntegrityError("v2 recovery identity is invalid")
                if record.recovery_processing_high_water != high_water:
                    raise EventJournalIntegrityError("recovery high-water is invalid")
                prior = recovery_open.get(record.recovery_id)
                if record.lifecycle is EventLifecycle.RECOVERY_PREPARED:
                    if prior is not None or record.recovery_id in seen_recovery_ids:
                        raise EventJournalIntegrityError("recovery is duplicated")
                    if record.recovery_category in {
                        EventRecoveryCategory.EXACT_CURRENT,
                        EventRecoveryCategory.UNCOMMITTED_TAIL,
                    } and (
                        record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "prepared current recovery semantics are invalid"
                        )
                    assert record.snapshot_sequence is not None
                    if (
                        record.recovery_category is EventRecoveryCategory.TRUE_ROLLBACK
                        and record.snapshot_sequence >= snapshot_sequence
                    ):
                        raise EventJournalIntegrityError(
                            "prepared rollback semantics are invalid"
                        )
                    recovery_open[record.recovery_id] = record
                    seen_recovery_ids.add(record.recovery_id)
                else:
                    if prior is None or (
                        prior.wal_generation_id != record.wal_generation_id
                        or (
                            prior.wal_record_id is not None
                            and (
                                prior.wal_record_id != record.wal_record_id
                                or prior.wal_record_hash != record.wal_record_hash
                            )
                        )
                        or prior.recovery_processing_high_water
                        != record.recovery_processing_high_water
                        or prior.snapshot_sequence != record.snapshot_sequence
                        or prior.snapshot_hash != record.snapshot_hash
                        or prior.recovery_category != record.recovery_category
                    ):
                        raise EventJournalIntegrityError("recovery pair is invalid")
                    if record.recovery_category in {
                        EventRecoveryCategory.EXACT_CURRENT,
                        EventRecoveryCategory.UNCOMMITTED_TAIL,
                    } and (
                        record.external_reconciliation_required
                        is not external_reconciliation_required
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "current recovery semantics are invalid"
                        )
                    assert record.snapshot_sequence is not None
                    if (
                        record.recovery_category is EventRecoveryCategory.TRUE_ROLLBACK
                        and (
                            record.external_reconciliation_required is not True
                            or record.snapshot_sequence >= snapshot_sequence
                        )
                    ):
                        raise EventJournalIntegrityError(
                            "rollback recovery semantics are invalid"
                        )
                    assert record.snapshot_sequence is not None
                    assert record.snapshot_hash is not None
                    snapshot_sequence = record.snapshot_sequence
                    snapshot_hash = record.snapshot_hash
                    external_reconciliation_required = bool(
                        record.external_reconciliation_required
                    )
                    wal_generation_id = record.wal_generation_id
                    wal_record_id = record.wal_record_id
                    wal_record_hash = record.wal_record_hash
                    wal_snapshot_sequence = record.snapshot_sequence
                    wal_snapshot_hash = record.snapshot_hash
                    completed_recoveries[record.recovery_id] = record
                    del recovery_open[record.recovery_id]
                continue
            if record.lifecycle in {
                EventLifecycle.STARTUP_RECONCILIATION_PREPARED,
                EventLifecycle.STARTUP_PARTICIPANT_RECONCILED,
                EventLifecycle.STARTUP_RECONCILIATION_COMPLETED,
            }:
                rid = record.reconciliation_id
                assert rid is not None
                if record.lifecycle is EventLifecycle.STARTUP_RECONCILIATION_PREPARED:
                    recovery = completed_recoveries.get(record.recovery_id or "")
                    recovery_binding = (
                        (
                            recovery.external_reconciliation_required,
                            recovery.snapshot_sequence,
                            recovery.snapshot_hash,
                            recovery.recovery_processing_high_water,
                            recovery.wal_generation_id,
                            recovery.wal_record_id,
                            recovery.wal_record_hash,
                        )
                        if recovery is not None
                        else (
                            external_reconciliation_required,
                            snapshot_sequence,
                            snapshot_hash,
                            high_water,
                            wal_generation_id,
                            wal_record_id,
                            wal_record_hash,
                        )
                    )
                    if (
                        recovery_binding
                        != (
                            True,
                            record.snapshot_sequence,
                            record.snapshot_hash,
                            record.recovery_processing_high_water,
                            record.wal_generation_id,
                            record.wal_record_id,
                            record.wal_record_hash,
                        )
                        or record.journal_lineage_id != journal_lineage_id
                    ):
                        raise EventJournalIntegrityError(
                            "startup reconciliation binding is invalid"
                        )
                    if (
                        rid in startup_seen_ids
                        or rid in startup_open
                        or record.recovery_id in startup_seen_recovery_ids
                    ):
                        raise EventJournalIntegrityError(
                            "startup reconciliation is duplicated"
                        )
                    assert record.required_participants is not None
                    startup_open[rid] = {
                        "record": record,
                        "required": record.required_participants,
                        "outcomes": {},
                        "completed": False,
                    }
                    startup_seen_ids.add(rid)
                    assert record.recovery_id is not None
                    startup_seen_recovery_ids.add(record.recovery_id)
                else:
                    state = startup_open.get(rid)
                    if state is None:
                        raise EventJournalIntegrityError(
                            "startup reconciliation is not open"
                        )
                    prior = state["record"]
                    common = (
                        record.recovery_id,
                        record.snapshot_sequence,
                        record.snapshot_hash,
                        record.recovery_processing_high_water,
                        record.wal_generation_id,
                        record.wal_record_id,
                        record.wal_record_hash,
                        record.journal_lineage_id,
                    )
                    expected = (
                        prior.recovery_id,
                        prior.snapshot_sequence,
                        prior.snapshot_hash,
                        prior.recovery_processing_high_water,
                        prior.wal_generation_id,
                        prior.wal_record_id,
                        prior.wal_record_hash,
                        prior.journal_lineage_id,
                    )
                    if common != expected:
                        raise EventJournalIntegrityError(
                            "startup reconciliation binding changed"
                        )
                    if (
                        record.lifecycle
                        is EventLifecycle.STARTUP_PARTICIPANT_RECONCILED
                    ):
                        assert record.participant_id is not None
                        assert record.operation_digest is not None
                        required = {
                            item.participant_id: item.operation_digest
                            for item in state["required"]
                        }
                        if (
                            record.participant_id not in required
                            or record.operation_digest
                            != required[record.participant_id]
                            or record.participant_id in state["outcomes"]
                        ):
                            raise EventJournalIntegrityError(
                                "startup participant evidence is invalid"
                            )
                        assert record.startup_participant_outcome is not None
                        state["outcomes"][record.participant_id] = (
                            record.operation_digest,
                            record.startup_participant_outcome,
                        )
                    else:
                        required_ids = {
                            item.participant_id for item in state["required"]
                        }
                        if set(state["outcomes"]) != required_ids:
                            raise EventJournalIntegrityError(
                                "startup reconciliation completion is premature"
                            )
                        state["completed"] = True
                        startup_completed.append(state)
                        del startup_open[rid]
                continue
            if record.lifecycle in {
                EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED,
                EventLifecycle.CLEAR_PREPARED,
                EventLifecycle.CLEARED,
            }:
                if record.lifecycle is EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED:
                    assert record.baseline_id is not None
                    assert record.participant_registry is not None
                    identifiers = tuple(
                        item.participant_id for item in record.participant_registry
                    )
                    if (
                        not identifiers
                        or identifiers != tuple(sorted(identifiers))
                        or len(set(identifiers)) != len(identifiers)
                        or record.baseline_id
                        in {item.baseline_id for item in baseline_records_seen}
                    ):
                        raise EventJournalIntegrityError(
                            "participant baseline authority is invalid"
                        )
                    current_epoch = record.adoption_epoch
                    if not baseline_records_seen:
                        if current_epoch not in {None, 0}:
                            raise EventJournalIntegrityError(
                                "participant baseline authority is invalid"
                            )
                    else:
                        previous = baseline_records_seen[-1]
                        previous_registry = {
                            item.participant_id: item.domain
                            for item in previous.participant_registry or ()
                        }
                        current_registry = {
                            item.participant_id: item.domain
                            for item in record.participant_registry
                        }
                        if (
                            current_epoch is None
                            or current_epoch != (previous.adoption_epoch or 0) + 1
                            or not set(current_registry).issuperset(previous_registry)
                            or set(current_registry) == set(previous_registry)
                            or any(
                                current_registry[item] != domain
                                for item, domain in previous_registry.items()
                            )
                        ):
                            raise EventJournalIntegrityError(
                                "participant baseline authority is invalid"
                            )
                    if (
                        record.schema_version < 3
                        or external_reconciliation_required
                        or open_events
                        or recovery_open
                        or any(
                            transaction["terminal"] is None
                            for transaction in transactions.values()
                        )
                        or startup_open
                        or gate_clear_open
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                        or record.recovery_processing_high_water != high_water
                        or record.wal_generation_id != wal_generation_id
                        or record.wal_record_id != wal_record_id
                        or record.wal_record_hash != wal_record_hash
                        or record.journal_lineage_id != journal_lineage_id
                    ):
                        raise EventJournalIntegrityError(
                            "participant baseline authority is invalid"
                        )
                    baseline_records_seen.append(record)
                elif record.lifecycle is EventLifecycle.CLEAR_PREPARED:
                    gate_clear_open = True
                else:
                    gate_clear_open = False
                    external_reconciliation_required = False
                continue
            if record.lifecycle in {
                EventLifecycle.TRANSACTION_PREPARED,
                EventLifecycle.PARTICIPANT_FINALIZED,
                EventLifecycle.TRANSACTION_COMPLETED,
                EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
                EventLifecycle.TRANSACTION_RECONCILED,
                EventLifecycle.PARTICIPANT_ABORTED,
                EventLifecycle.TRANSACTION_ABORT_REQUIRED,
                EventLifecycle.TRANSACTION_ABORTED,
            }:
                if (
                    record.schema_version < 3
                    or record.event_id is None
                    or record.event_type is None
                    or record.source is None
                    or record.transaction_id is None
                ):
                    raise EventJournalIntegrityError("transaction identity is invalid")
                current = open_events.get(record.event_id)
                if (
                    current is None
                    or record.processing_sequence != current.processing_sequence
                ):
                    raise EventJournalIntegrityError(
                        "transaction event anchor is invalid"
                    )
                tx = transactions.get(record.transaction_id)
                if record.lifecycle is EventLifecycle.TRANSACTION_PREPARED:
                    if (
                        tx is not None
                        or current.lifecycle is not EventLifecycle.STARTED
                        or record.transaction_kind is None
                        or record.required_participants is None
                    ):
                        raise EventJournalIntegrityError(
                            "transaction preparation is invalid"
                        )
                    transactions[record.transaction_id] = {
                        "event_id": record.event_id,
                        "event_type": record.event_type,
                        "source": record.source,
                        "sequence": record.processing_sequence,
                        "kind": record.transaction_kind,
                        "required": record.required_participants,
                        "outcomes": {},
                        "reason": None,
                        "unresolved": (),
                        "abort_outcomes": {},
                        "abort_reason": None,
                        "branch": None,
                        "terminal": None,
                    }
                else:
                    if (
                        tx is None
                        or current.lifecycle
                        not in {EventLifecycle.PREPARED, EventLifecycle.STARTED}
                        or (
                            tx["event_id"],
                            tx["event_type"],
                            tx["source"],
                            tx["sequence"],
                        )
                        != (
                            record.event_id,
                            record.event_type,
                            record.source,
                            record.processing_sequence,
                        )
                    ):
                        raise EventJournalIntegrityError("transaction identity changed")
                    required = {
                        item.participant_id: item.operation_digest
                        for item in tx["required"]
                    }
                    if (
                        record.lifecycle
                        in {
                            EventLifecycle.PARTICIPANT_FINALIZED,
                            EventLifecycle.TRANSACTION_COMPLETED,
                            EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
                            EventLifecycle.TRANSACTION_RECONCILED,
                        }
                        and current.lifecycle is not EventLifecycle.PREPARED
                    ):
                        raise EventJournalIntegrityError(
                            "finalization evidence is before internal prepare"
                        )
                    if (
                        record.lifecycle
                        in {
                            EventLifecycle.PARTICIPANT_ABORTED,
                            EventLifecycle.TRANSACTION_ABORT_REQUIRED,
                            EventLifecycle.TRANSACTION_ABORTED,
                        }
                        and current.lifecycle
                        not in {EventLifecycle.STARTED, EventLifecycle.PREPARED}
                    ):
                        raise EventJournalIntegrityError(
                            "abort evidence has no startup-eligible event state"
                        )
                    if record.lifecycle is EventLifecycle.PARTICIPANT_FINALIZED:
                        if tx["branch"] == "abort":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "finalize"
                        if (
                            record.participant_id is None
                            or record.operation_digest
                            != required.get(record.participant_id)
                            or record.participant_id in tx["outcomes"]
                            or tx["terminal"] is not None
                        ):
                            raise EventJournalIntegrityError(
                                "participant finalization is invalid"
                            )
                        assert record.participant_outcome is not None
                        tx["outcomes"][record.participant_id] = (
                            record.participant_outcome
                        )
                        if tx["reason"] is not None:
                            tx["unresolved"] = tuple(
                                sorted(set(required) - set(tx["outcomes"]))
                            )
                    elif (
                        record.lifecycle
                        is EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED
                    ):
                        if tx["branch"] == "abort":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "finalize"
                        missing = tuple(sorted(set(required) - set(tx["outcomes"])))
                        if (
                            tx["reason"] is not None
                            or tuple(record.unresolved_participants or ()) != missing
                            or not missing
                        ):
                            raise EventJournalIntegrityError(
                                "transaction reconciliation is invalid"
                            )
                        tx["reason"] = record.reconciliation_reason
                        tx["unresolved"] = missing
                    elif record.lifecycle is EventLifecycle.TRANSACTION_COMPLETED:
                        if (
                            tx["branch"] == "abort"
                            or tx["terminal"] is not None
                            or tx["reason"] is not None
                            or set(tx["outcomes"]) != set(required)
                        ):
                            raise EventJournalIntegrityError(
                                "transaction completion is premature"
                            )
                        tx["terminal"] = EventLifecycle.TRANSACTION_COMPLETED
                    elif record.lifecycle is EventLifecycle.PARTICIPANT_ABORTED:
                        if tx["branch"] == "finalize":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "abort"
                        requirements = {
                            item.participant_id: item for item in tx["required"]
                        }
                        requirement = requirements.get(record.participant_id or "")
                        if (
                            requirement is None
                            or ParticipantCapability.ABORT
                            not in requirement.capabilities
                            or record.operation_digest != requirement.operation_digest
                            or record.participant_id in tx["abort_outcomes"]
                            or tx["terminal"] is not None
                        ):
                            raise EventJournalIntegrityError(
                                "participant abort is invalid"
                            )
                        assert record.abort_outcome is not None
                        tx["abort_outcomes"][record.participant_id] = (
                            record.abort_outcome
                        )
                        if tx["abort_reason"] is not None:
                            tx["unresolved"] = tuple(
                                sorted(
                                    participant
                                    for participant, item in requirements.items()
                                    if ParticipantCapability.ABORT in item.capabilities
                                    and participant not in tx["abort_outcomes"]
                                )
                            )
                    elif record.lifecycle is EventLifecycle.TRANSACTION_ABORT_REQUIRED:
                        if tx["branch"] == "finalize":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "abort"
                        requirements = {
                            item.participant_id: item for item in tx["required"]
                        }
                        missing = tuple(
                            sorted(
                                participant
                                for participant, item in requirements.items()
                                if ParticipantCapability.ABORT in item.capabilities
                                and participant not in tx["abort_outcomes"]
                            )
                        )
                        if (
                            tx["abort_reason"] is not None
                            or tuple(record.unresolved_participants or ()) != missing
                            or not missing
                        ):
                            raise EventJournalIntegrityError(
                                "transaction abort is invalid"
                            )
                        tx["abort_reason"] = record.abort_reason
                        tx["unresolved"] = missing
                    elif record.lifecycle is EventLifecycle.TRANSACTION_ABORTED:
                        abort_ids = {
                            item.participant_id
                            for item in tx["required"]
                            if ParticipantCapability.ABORT in item.capabilities
                        }
                        if (
                            tx["terminal"] is not None
                            or set(tx["abort_outcomes"]) != abort_ids
                            or (abort_ids and tx["branch"] != "abort")
                            or (not abort_ids and tx["branch"] is not None)
                        ):
                            raise EventJournalIntegrityError(
                                "transaction abort is incomplete"
                            )
                        tx["unresolved"] = ()
                        tx["terminal"] = EventLifecycle.TRANSACTION_ABORTED
                    elif (
                        tx["reason"] is None
                        or set(tx["outcomes"]) != set(required)
                        or tx["terminal"] is not None
                        or tx["branch"] == "abort"
                    ):
                        raise EventJournalIntegrityError(
                            "transaction reconciliation is incomplete"
                        )
                    elif tx["branch"] != "abort":
                        tx["terminal"] = EventLifecycle.TRANSACTION_RECONCILED
                    else:
                        raise EventJournalIntegrityError(
                            "transaction branch is invalid"
                        )
                continue
            assert record.event_id is not None
            assert record.event_type is not None
            assert record.source is not None
            current = open_events.get(record.event_id)
            if record.lifecycle is EventLifecycle.ACCEPTED:
                if record.event_id in seen_event_ids:
                    raise EventJournalIntegrityError("Event identifier is duplicated")
                seen_event_ids.add(record.event_id)
                open_events[record.event_id] = _EventState(
                    record.event_id,
                    record.event_type,
                    record.source,
                    record.timestamp,
                    EventLifecycle.ACCEPTED,
                    None,
                )
                accepted_queue.append(record.event_id)
                continue
            if current is None:
                raise EventJournalIntegrityError("Event lifecycle lacks acceptance")
            if (record.event_type, record.source) != (
                current.event_type,
                current.source,
            ):
                raise EventJournalIntegrityError("Event metadata changed")

            if record.lifecycle is EventLifecycle.STARTED:
                if (
                    current.lifecycle is not EventLifecycle.ACCEPTED
                    or processing_event_id is not None
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                ):
                    raise EventJournalIntegrityError("Started lifecycle is impossible")
                if high_water >= MAX_PERSISTED_EVENT_SEQUENCE or (
                    record.processing_sequence != high_water + 1
                ):
                    raise EventJournalIntegrityError("Processing sequence has a gap")
                high_water = record.processing_sequence
                processing_event_id = record.event_id
                open_events[record.event_id] = _EventState(
                    current.event_id,
                    current.event_type,
                    current.source,
                    current.requested_at,
                    EventLifecycle.STARTED,
                    record.processing_sequence,
                )
            elif record.lifecycle is EventLifecycle.PREPARED:
                if (
                    processing_event_id != record.event_id
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                    or current.lifecycle is not EventLifecycle.STARTED
                    or record.processing_sequence != current.processing_sequence
                    or record.state_hash_before != snapshot_hash
                    or (
                        # Since a rollover moves the WAL anchor at ordinary
                        # boundaries, a v4 event must bind the current
                        # generation, never an older one (IP-3a review P1-1).
                        record.schema_version >= 4
                        and record.wal_generation_id != wal_generation_id
                    )
                ):
                    raise EventJournalIntegrityError("Prepared lifecycle is impossible")
                if any(
                    transaction["event_id"] == record.event_id
                    and transaction["terminal"] is EventLifecycle.TRANSACTION_ABORTED
                    for transaction in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "aborted transaction cannot cross internal prepare"
                    )
                open_events[record.event_id] = _EventState(
                    current.event_id,
                    current.event_type,
                    current.source,
                    current.requested_at,
                    EventLifecycle.PREPARED,
                    current.processing_sequence,
                    record.state_hash_before,
                    record.state_hash_after,
                    record.wal_generation_id,
                )
            elif record.lifecycle is EventLifecycle.COMPLETED:
                if (
                    processing_event_id != record.event_id
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                    or current.lifecycle is not EventLifecycle.PREPARED
                    or record.processing_sequence != current.processing_sequence
                    or record.snapshot_sequence != current.processing_sequence
                    or record.snapshot_hash != current.state_hash_after
                    or (
                        record.schema_version >= 2
                        and record.wal_generation_id != current.wal_generation_id
                    )
                ):
                    raise EventJournalIntegrityError(
                        "Completed lifecycle is impossible"
                    )
                if any(
                    t["event_id"] == record.event_id and t["terminal"] is None
                    for t in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "event completed with open transaction evidence"
                    )
                assert record.snapshot_sequence is not None
                assert record.snapshot_hash is not None
                snapshot_sequence = record.snapshot_sequence
                snapshot_hash = record.snapshot_hash
                wal_generation_id = record.wal_generation_id
                wal_record_id = record.wal_record_id
                wal_record_hash = record.wal_record_hash
                wal_snapshot_sequence = record.snapshot_sequence
                wal_snapshot_hash = record.snapshot_hash
                del open_events[record.event_id]
                accepted_queue.pop(0)
                processing_event_id = None
            elif record.lifecycle is EventLifecycle.FAILED:
                if (
                    processing_event_id != record.event_id
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                    or current.lifecycle is not EventLifecycle.STARTED
                    or record.processing_sequence != current.processing_sequence
                    or record.snapshot_sequence != snapshot_sequence
                    or record.snapshot_hash != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Failed lifecycle is impossible")
                if any(
                    t["event_id"] == record.event_id and t["terminal"] is None
                    for t in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "event failed with open transaction evidence"
                    )
                del open_events[record.event_id]
                accepted_queue.pop(0)
                processing_event_id = None
            elif record.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED:
                if record.snapshot_sequence is None or record.snapshot_hash is None:
                    raise EventJournalIntegrityError("Recovery snapshot is missing")
                if any(
                    transaction["event_id"] == record.event_id
                    and transaction["terminal"] is None
                    for transaction in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "recovery cannot hide open transaction evidence"
                    )
                if record.failure_category is EventFailureCategory.ACCEPTED_NOT_STARTED:
                    if (
                        current.lifecycle is not EventLifecycle.ACCEPTED
                        or processing_event_id is not None
                        or not accepted_queue
                        or accepted_queue[0] != record.event_id
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "Accepted recovery is impossible"
                        )
                elif (
                    record.failure_category
                    is EventFailureCategory.UNCOMMITTED_AFTER_CRASH
                ):
                    if (
                        processing_event_id != record.event_id
                        or not accepted_queue
                        or accepted_queue[0] != record.event_id
                        or current.lifecycle
                        not in {
                            EventLifecycle.STARTED,
                            EventLifecycle.PREPARED,
                        }
                        or record.processing_sequence != current.processing_sequence
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "Uncommitted recovery is impossible"
                        )
                elif (
                    record.failure_category
                    is EventFailureCategory.COMMITTED_BEFORE_CRASH
                ):
                    if (
                        processing_event_id != record.event_id
                        or not accepted_queue
                        or accepted_queue[0] != record.event_id
                        or current.lifecycle is not EventLifecycle.PREPARED
                        or record.processing_sequence != current.processing_sequence
                        or record.snapshot_sequence != current.processing_sequence
                        or record.snapshot_hash != current.state_hash_after
                    ):
                        raise EventJournalIntegrityError(
                            "Committed recovery is impossible"
                        )
                    snapshot_sequence = record.snapshot_sequence
                    snapshot_hash = record.snapshot_hash
                else:
                    raise EventJournalIntegrityError("Recovery category is impossible")
                del open_events[record.event_id]
                accepted_queue.pop(0)
                processing_event_id = None
            else:
                raise EventJournalIntegrityError("Lifecycle is impossible")

        verifier.high_water = high_water
        verifier.snapshot_sequence = snapshot_sequence
        verifier.snapshot_hash = snapshot_hash
        verifier.journal_lineage_id = journal_lineage_id
        verifier.wal_generation_id = wal_generation_id
        verifier.wal_record_id = wal_record_id
        verifier.wal_record_hash = wal_record_hash
        verifier.wal_snapshot_sequence = wal_snapshot_sequence
        verifier.wal_snapshot_hash = wal_snapshot_hash
        verifier.external_reconciliation_required = external_reconciliation_required
        verifier.v2_seen = v2_seen
        verifier.v3_seen = v3_seen
        verifier.v3_migration_anchor_hash = v3_migration_anchor_hash
        verifier.v4_seen = v4_seen
        verifier.v4_migration_anchor_hash = v4_migration_anchor_hash
        verifier.rollover_open = rollover_open
        verifier.last_compaction_hash = last_compaction_hash
        verifier.injected_compaction_passed = injected_compaction_passed
        verifier.processing_event_id = processing_event_id
        verifier.gate_clear_open = gate_clear_open

        views: list[EventJournalTransaction] = []
        for transaction_id, value in transactions.items():
            transaction_view = view_cache.get(transaction_id)
            if transaction_view is None:
                transaction_view = EventJournalTransaction(
                    transaction_id=transaction_id,
                    event_id=value["event_id"],
                    event_type=value["event_type"],
                    source=value["source"],
                    processing_sequence=value["sequence"],
                    kind=value["kind"],
                    required_participants=value["required"],
                    participant_outcomes=tuple(sorted(value["outcomes"].items())),
                    abort_outcomes=tuple(sorted(value["abort_outcomes"].items())),
                    reconciliation_reason=value["reason"],
                    abort_reason=value["abort_reason"],
                    unresolved_participants=value["unresolved"],
                    terminal_lifecycle=value["terminal"],
                )
                view_cache[transaction_id] = transaction_view
            views.append(transaction_view)
        transaction_views = tuple(views)
        startup_views = tuple(
            EventJournalStartupReconciliation(
                reconciliation_id=rid,
                recovery_id=state["record"].recovery_id or "",
                snapshot_sequence=state["record"].snapshot_sequence or 0,
                snapshot_hash=state["record"].snapshot_hash or "0" * 64,
                recovery_processing_high_water=(
                    state["record"].recovery_processing_high_water or 0
                ),
                wal_generation_id=state["record"].wal_generation_id or "",
                wal_record_id=state["record"].wal_record_id or "",
                wal_record_hash=state["record"].wal_record_hash or "0" * 64,
                journal_lineage_id=state["record"].journal_lineage_id or "",
                required_participants=state["required"],
                participant_outcomes=tuple(
                    (participant, digest, outcome)
                    for participant, (digest, outcome) in sorted(
                        state["outcomes"].items()
                    )
                ),
                completed=state["completed"],
            )
            for rid, state in [
                *startup_open.items(),
                *[
                    (item["record"].reconciliation_id, item)
                    for item in startup_completed
                ],
            ]
        )
        baseline_views: list[EventJournalParticipantBaseline] = []
        baseline_records: dict[str, EventJournalRecord] = {}
        for record in gate_records:
            if record.lifecycle is EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED:
                assert record.baseline_id is not None
                baseline_records[record.baseline_id] = record
                assert record.participant_registry is not None
                baseline_views.append(EventJournalParticipantBaseline(
                    record.baseline_id, record.snapshot_sequence or 0,
                    record.snapshot_hash or "0" * 64,
                    record.recovery_processing_high_water or 0,
                    record.wal_generation_id or "", record.wal_record_id or "",
                    record.wal_record_hash or "0" * 64,
                    record.journal_lineage_id or "", record.participant_registry,
                    record.adoption_epoch or 0,
                ))
        open_gate: EventJournalGateClear | None = None
        terminal_gate: EventJournalGateClear | None = None
        for record in gate_records:
            if record.lifecycle not in {
                EventLifecycle.CLEAR_PREPARED, EventLifecycle.CLEARED
            }:
                continue
            assert record.baseline_id is not None
            baseline = baseline_records.get(record.baseline_id)
            if baseline is None:
                raise EventJournalIntegrityError("gate clear baseline is unknown")
            if (
                record.snapshot_sequence is None
                or baseline.snapshot_sequence is None
                or record.snapshot_sequence < baseline.snapshot_sequence
                or record.recovery_processing_high_water is None
                or baseline.recovery_processing_high_water is None
                or record.recovery_processing_high_water
                < baseline.recovery_processing_high_water
                or record.journal_lineage_id != baseline.journal_lineage_id
            ):
                raise EventJournalIntegrityError("gate clear baseline coverage is invalid")
            startup = next(
                (item for item in startup_views
                 if item.reconciliation_id == (record.reconciliation_id or "")
                 and item.recovery_id == (record.recovery_id or "")
                 and item.completed),
                None,
            )
            if startup is None or (
                startup.snapshot_sequence != record.snapshot_sequence
                or startup.snapshot_hash != record.snapshot_hash
                or startup.recovery_processing_high_water
                != record.recovery_processing_high_water
                or startup.wal_generation_id != record.wal_generation_id
                or startup.wal_record_id != record.wal_record_id
                or startup.wal_record_hash != record.wal_record_hash
                or startup.journal_lineage_id != record.journal_lineage_id
            ):
                raise EventJournalIntegrityError("gate clear reconciliation binding is invalid")
            view = EventJournalGateClear(
                record.baseline_id, record.reconciliation_id or "",
                record.recovery_id or "", record.snapshot_sequence or 0,
                record.snapshot_hash or "0" * 64,
                record.recovery_processing_high_water or 0,
                record.wal_generation_id or "", record.wal_record_id or "",
                record.wal_record_hash or "0" * 64,
                record.journal_lineage_id or "",
                record.lifecycle is EventLifecycle.CLEARED,
            )
            if record.lifecycle is EventLifecycle.CLEAR_PREPARED:
                if open_gate is not None:
                    raise EventJournalIntegrityError("gate clear is duplicated")
                if (
                    terminal_gate is not None
                    and terminal_gate.reconciliation_id == view.reconciliation_id
                ):
                    raise EventJournalIntegrityError("gate clear is duplicated")
                open_gate = view
                terminal_gate = None
            else:
                if open_gate is None or terminal_gate is not None:
                    raise EventJournalIntegrityError("gate clear completion is impossible")
                if view != EventJournalGateClear(
                    open_gate.baseline_id, open_gate.reconciliation_id,
                    open_gate.recovery_id, open_gate.snapshot_sequence,
                    open_gate.snapshot_hash, open_gate.processing_high_water,
                    open_gate.wal_generation_id, open_gate.wal_record_id,
                    open_gate.wal_record_hash, open_gate.journal_lineage_id, True
                ):
                    raise EventJournalIntegrityError("gate clear completion binding changed")
                terminal_gate = view
                open_gate = None
        if terminal_gate is not None:
            external_reconciliation_required = False
        result = _VerifiedJournal(
            records,
            high_water,
            snapshot_sequence,
            snapshot_hash,
            tuple(open_events.values()),
            journal_lineage_id,
            external_reconciliation_required,
            wal_snapshot_sequence,
            wal_snapshot_hash,
            tuple(recovery_open.values()),
            wal_generation_id,
            wal_record_id,
            wal_record_hash,
            transaction_views,
            startup_views,
            tuple(baseline_views),
            open_gate,
            terminal_gate,
            (
                None
                if rollover_open is None
                else EventJournal._rollover_view(rollover_open)
            ),
            EventJournal._latest_compaction(records),
            (
                ()
                if injected_compaction is None
                or injected_compaction.compaction is None
                else injected_compaction.compaction.carried_records
            ),
        )
        verifier.records = records
        verifier.result = result
        return result, verifier

    @staticmethod
    def _initial_verifier_state(
        records: tuple[EventJournalRecord, ...],
    ) -> _VerifierState:
        if (
            not records
            or records[0].lifecycle is not EventLifecycle.CHECKPOINT
            or records[0].schema_version not in {1, 2, 3, 4}
        ):
            raise EventJournalIntegrityError("EventJournal lacks checkpoint authority")
        checkpoint = records[0]
        assert checkpoint.processing_sequence is not None
        assert checkpoint.snapshot_sequence is not None
        assert checkpoint.snapshot_hash is not None
        high_water = checkpoint.processing_sequence
        snapshot_sequence = checkpoint.snapshot_sequence
        snapshot_hash = checkpoint.snapshot_hash
        if (
            type(high_water) is not int
            or type(snapshot_sequence) is not int
            or high_water > MAX_PERSISTED_EVENT_SEQUENCE
            or snapshot_sequence > MAX_PERSISTED_EVENT_SEQUENCE
        ):
            raise EventJournalIntegrityError(
                "EventJournal processing sequence is outside its bound"
            )
        journal_lineage_id = checkpoint.journal_lineage_id
        wal_generation_id = checkpoint.wal_generation_id
        wal_record_id = checkpoint.wal_record_id
        wal_record_hash = checkpoint.wal_record_hash
        wal_snapshot_sequence = snapshot_sequence
        wal_snapshot_hash = snapshot_hash
        external_reconciliation_required = bool(
            checkpoint.external_reconciliation_required
        )
        if checkpoint.schema_version >= 2 and (
            checkpoint.wal_generation_id is None
            or checkpoint.wal_record_id is None
            or checkpoint.wal_record_hash is None
            or checkpoint.journal_lineage_id is None
            or checkpoint.migration_previous_v1_hash is not None
            or checkpoint.migration_previous_v2_hash is not None
            or checkpoint.migration_previous_v3_hash is not None
        ):
            raise EventJournalIntegrityError("WAL-bound checkpoint anchor is invalid")
        if snapshot_sequence > high_water:
            raise EventJournalIntegrityError("Checkpoint sequence is impossible")
        state = _VerifierState(
            records=(),
            high_water=high_water,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            journal_lineage_id=journal_lineage_id,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
            wal_snapshot_sequence=wal_snapshot_sequence,
            wal_snapshot_hash=wal_snapshot_hash,
            external_reconciliation_required=external_reconciliation_required,
            v2_seen=checkpoint.schema_version == 2,
            v3_seen=checkpoint.schema_version >= 3,
            v3_migration_anchor_hash=checkpoint.v3_migration_anchor_hash,
            v4_seen=checkpoint.schema_version == 4,
            v4_migration_anchor_hash=checkpoint.v4_migration_anchor_hash,
        )
        EventJournal._inject_compaction(records, state)
        return state

    @staticmethod
    def _latest_compaction(
        records: tuple[EventJournalRecord, ...],
    ) -> EventJournalCompaction | None:
        latest = next(
            (
                record
                for record in reversed(records)
                if record.lifecycle is EventLifecycle.COMPACTION
            ),
            None,
        )
        return None if latest is None else latest.compaction

    @staticmethod
    def _inject_compaction(
        records: tuple[EventJournalRecord, ...], state: _VerifierState
    ) -> None:
        """Start replay from the state of the compaction IN EFFECT.

        The compaction in effect is the one whose boundary checkpoint is the
        oldest retained record (its archived segments are gone). A newer
        compaction whose archived segments are still present (being appended,
        or unlinks pending) is checked against those records during replay,
        starting from this injected state. Several compactions may be
        retained; the C7 chain starts at the oldest retained one, whose
        predecessor may itself be archived (IP-4a review P0-1, P0-2).
        """
        compactions = [
            record
            for record in records
            if record.lifecycle is EventLifecycle.COMPACTION
        ]
        if not compactions:
            return
        oldest = compactions[0]
        assert oldest.compaction is not None
        state.last_compaction_hash = oldest.compaction.previous_compaction_hash
        effective = next(
            (
                record
                for record in compactions
                if record.compaction is not None
                and record.compaction.boundary_checkpoint_hash
                == records[0].record_hash
            ),
            None,
        )
        if effective is None:
            return
        assert effective.compaction is not None
        declared = effective.compaction
        # Chain connection (#381 D1 P1-2): the oldest retained checkpoint must
        # continue exactly where the archived range ended.
        if records[0].previous_record_hash != declared.archived_last_record_hash:
            raise EventJournalIntegrityError("compaction chain is not connected")
        for carried in declared.carried_records:
            if EventJournal._record_hash(carried) != carried.record_hash:
                raise EventJournalIntegrityError(
                    "compaction carried record is not authentic"
                )
        state.injected_compaction = effective
        if declared.baseline_record is not None:
            state.baseline_records_seen.append(declared.baseline_record)
            state.gate_records.append(declared.baseline_record)

    @staticmethod
    def _archived_root(records: tuple[EventJournalRecord, ...]) -> str:
        digest = hashlib.sha256(_ARCHIVE_ROOT_DOMAIN)
        for record in records:
            digest.update(bytes.fromhex(record.record_hash))
        return digest.hexdigest()

    @staticmethod
    def _declared_state_at(
        records: tuple[EventJournalRecord, ...],
        boundary: int,
        injected: EventJournalRecord | None,
    ) -> tuple[
        EventJournalRecord | None,
        EventJournalRecord | None,
        EventJournalRecord | None,
    ]:
        """Latest baseline and true-rollback pair at ``records[boundary]``."""
        prior = (
            ()
            if injected is None or injected.compaction is None
            else injected.compaction.carried_records
        )
        baseline = next(
            (
                record
                for record in reversed(records[:boundary])
                if record.lifecycle
                is EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED
            ),
            next(
                (
                    record
                    for record in prior
                    if record.lifecycle
                    is EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED
                ),
                None,
            ),
        )
        rollback: EventJournalRecord | None = None
        cleared: EventJournalRecord | None = None
        for position in range(boundary - 1, -1, -1):
            candidate = records[position]
            if (
                candidate.lifecycle is EventLifecycle.RECOVERY_COMPLETED
                and candidate.recovery_category is EventRecoveryCategory.TRUE_ROLLBACK
                and candidate.external_reconciliation_required is True
            ):
                rollback = candidate
                cleared = next(
                    (
                        later
                        for later in records[position + 1 : boundary]
                        if later.lifecycle is EventLifecycle.CLEARED
                    ),
                    None,
                )
                break
        if rollback is None:
            rollback = next(
                (
                    record
                    for record in prior
                    if record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
                ),
                None,
            )
            cleared = next(
                (record for record in prior if record.lifecycle is EventLifecycle.CLEARED),
                None,
            )
        if rollback is None or cleared is None:
            rollback = cleared = None
        return baseline, rollback, cleared

    @staticmethod
    def _verify_declared_compaction(
        records: tuple[EventJournalRecord, ...],
        boundary: int,
        declared: EventJournalCompaction,
        injected: EventJournalRecord | None,
    ) -> None:
        """Check a compaction against the archived records still present."""
        archived = records[:boundary]
        boundary_record = records[boundary]
        if (
            boundary_record.lifecycle is not EventLifecycle.CHECKPOINT
            or boundary_record.processing_sequence is None
            or archived[0].processing_sequence is None
            or declared.archived_first_record_hash != archived[0].record_hash
            or declared.archived_last_record_hash != archived[-1].record_hash
            or declared.archived_record_count != len(archived)
            or declared.archived_root != EventJournal._archived_root(archived)
            or declared.archived_processing_floor != archived[0].processing_sequence
            or declared.archived_processing_high_water
            != boundary_record.processing_sequence
            or declared.archived_first_timestamp != archived[0].timestamp
            or declared.archived_last_timestamp != archived[-1].timestamp
        ):
            raise EventJournalIntegrityError("compaction archived range is invalid")
        baseline, rollback, cleared = EventJournal._declared_state_at(
            records, boundary, injected
        )
        if (
            declared.baseline_record != baseline
            or declared.true_rollback_completed_record != rollback
            or declared.true_rollback_cleared_record != cleared
        ):
            raise EventJournalIntegrityError("compaction declared state is invalid")
        tail = declared.carried_tail_record
        prior = (
            ()
            if injected is None or injected.compaction is None
            else injected.compaction.carried_records
        )
        if tail is not None and tail not in archived and tail not in prior:
            raise EventJournalIntegrityError("compaction carried tail is unknown")

    @staticmethod
    def _rollover_view(record: EventJournalRecord) -> EventJournalWalRollover:
        assert record.rollover_id is not None
        assert record.snapshot_sequence is not None
        assert record.snapshot_hash is not None
        assert record.wal_generation_id is not None
        assert record.wal_record_id is not None
        assert record.wal_record_hash is not None
        assert record.wal_predecessor_generation_id is not None
        assert record.wal_predecessor_record_hash is not None
        return EventJournalWalRollover(
            rollover_id=record.rollover_id,
            snapshot_sequence=record.snapshot_sequence,
            snapshot_hash=record.snapshot_hash,
            wal_generation_id=record.wal_generation_id,
            wal_record_id=record.wal_record_id,
            wal_record_hash=record.wal_record_hash,
            wal_predecessor_generation_id=record.wal_predecessor_generation_id,
            wal_predecessor_record_hash=record.wal_predecessor_record_hash,
        )

    @staticmethod
    def _migration_anchors(
        records: tuple[EventJournalRecord, ...], schema_version: int
    ) -> tuple[str | None, str | None]:
        """The v3 and v4 migration authorities a new checkpoint must carry."""
        v3_anchor: str | None = None
        v4_anchor: str | None = None
        if schema_version >= 3:
            v3_anchor = next(
                (
                    record.v3_migration_anchor_hash
                    for record in reversed(records)
                    if record.schema_version >= 3
                    and record.lifecycle is EventLifecycle.CHECKPOINT
                    and record.v3_migration_anchor_hash is not None
                ),
                None,
            )
            if v3_anchor is None:
                raise EventJournalIntegrityError("v3 Journal lacks migration authority")
        if schema_version >= 4:
            v4_anchor = next(
                (
                    record.v4_migration_anchor_hash
                    for record in reversed(records)
                    if record.schema_version >= 4
                    and record.lifecycle is EventLifecycle.CHECKPOINT
                    and record.v4_migration_anchor_hash is not None
                ),
                None,
            )
            if v4_anchor is None:
                raise EventJournalIntegrityError("v4 Journal lacks migration authority")
        return v3_anchor, v4_anchor

    def _rotated_paths_unlocked(self) -> list[tuple[int, Path]]:
        return [
            (index, path)
            for index, path in self._journal_artifacts_unlocked()
            if index is not None
        ]

    def _segment_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index:08d}")

    def _fsync_parent(self) -> None:
        descriptor = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _run_hook(
        self, lifecycle: EventLifecycle, stage: EventJournalAppendStage
    ) -> None:
        if self._append_stage_hook is not None:
            self._append_stage_hook(lifecycle, stage)

    def _require_authority(self) -> None:
        if not self._lease.held:
            raise EventJournalLoadError(
                "EventJournal exclusive authority is unavailable"
            )

    @staticmethod
    def _validate_snapshot_identity(sequence: int, snapshot_hash: str) -> None:
        if (
            type(sequence) is not int
            or sequence < 0
            or sequence > MAX_PERSISTED_EVENT_SEQUENCE
            or not isinstance(snapshot_hash, str)
            or _HASH_PATTERN.fullmatch(snapshot_hash) is None
        ):
            raise ValueError("canonical snapshot identity is invalid")

    @staticmethod
    def _reject_json_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")
