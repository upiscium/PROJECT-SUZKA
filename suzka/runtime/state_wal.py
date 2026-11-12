"""Durable, non-authoritative storage for reconstructible agent state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE
from suzka.privacy import normalize_private_key
from suzka.runtime.agent_state import (
    AgentStateStore,
    CompatibleAgentStateSnapshot,
    validate_compatible_agent_state_snapshot,
)
from suzka.runtime.attention_state_codec import preflight_attention_json


class StateWALError(Exception):
    """Bounded base class for WAL failures."""


class StateWALMissing(StateWALError):
    """The WAL has not been bootstrapped."""


class StateWALFormatError(StateWALError):
    """A WAL artifact is malformed or unsafe."""


class StateWALIntegrityError(StateWALError):
    """A WAL hash, lineage, or manifest constraint is invalid."""


class StateWALPermissionError(StateWALError):
    """A WAL artifact has unsafe permissions or ownership."""


class StateWALConflictError(StateWALError):
    """A requested operation conflicts with retained WAL state."""


class StateWALGateClearProofError(StateWALConflictError):
    """The proof-bound external reconciliation gate clear is stale."""


class RecoveryReason(str, Enum):
    BOOTSTRAP = "bootstrap"
    EXACT_CURRENT_REPAIR = "exact_current_repair"
    TRUE_ROLLBACK = "true_rollback"
    UNCOMMITTED_TAIL = "uncommitted_tail"
    # Ordinary generation switch at a clean boundary (#382 A): the baseline is
    # the latest committed snapshot, the state is unchanged.
    ROLLOVER = "rollover"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BaselineRecord(_StrictModel):
    record_type: Literal["baseline"]
    schema_version: Literal[1]
    command_version: Literal[1]
    record_id: UUID
    generation_id: UUID
    created_at: datetime
    baseline_snapshot_sequence: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    baseline_snapshot_hash: str = Field(min_length=64, max_length=64)
    baseline_snapshot: CompatibleAgentStateSnapshot
    journal_processing_high_water: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    predecessor_generation_id: UUID | None = None
    predecessor_generation_hash: str | None = None
    reason: RecoveryReason
    previous_record_hash: str | None
    record_hash: str = Field(min_length=64, max_length=64)

    @field_validator(
        "baseline_snapshot_sequence", "journal_processing_high_water", mode="before"
    )
    @classmethod
    def require_exact_persisted_sequence(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("persisted processing sequence must be an integer")
        return value


class TransitionRecord(_StrictModel):
    record_type: Literal["transition"]
    schema_version: Literal[1]
    command_version: Literal[1]
    record_id: UUID
    generation_id: UUID
    created_at: datetime
    event_id: UUID
    event_type: str = Field(min_length=1, max_length=128)
    event_source: str = Field(min_length=1, max_length=128)
    processing_sequence: int = Field(ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE)
    prior_snapshot_sequence: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    prior_snapshot_hash: str = Field(min_length=64, max_length=64)
    candidate_snapshot_sequence: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    candidate_snapshot_hash: str = Field(min_length=64, max_length=64)
    candidate_snapshot: CompatibleAgentStateSnapshot
    previous_record_hash: str = Field(min_length=64, max_length=64)
    record_hash: str = Field(min_length=64, max_length=64)

    @field_validator(
        "processing_sequence",
        "prior_snapshot_sequence",
        "candidate_snapshot_sequence",
        mode="before",
    )
    @classmethod
    def require_exact_persisted_sequence(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("persisted processing sequence must be an integer")
        return value


class Manifest(_StrictModel):
    schema_version: Literal[1]
    command_version: Literal[1]
    active_generation_id: UUID
    active_baseline_record_id: UUID
    active_baseline_record_hash: str = Field(min_length=64, max_length=64)
    predecessor_generation_id: UUID | None
    predecessor_generation_hash: str | None
    external_reconciliation_required: bool
    manifest_hash: str = Field(min_length=64, max_length=64)


class BootAnchor(_StrictModel):
    schema_version: Literal[1]
    command_version: Literal[1]
    snapshot_sequence: int = Field(ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE)
    snapshot_hash: str = Field(min_length=64, max_length=64)
    generation_id: UUID
    anchored_record_id: UUID
    anchored_record_hash: str = Field(min_length=64, max_length=64)
    journal_processing_high_water: int = Field(
        ge=0, le=MAX_PERSISTED_EVENT_SEQUENCE
    )
    journal_tail_record_id: UUID | None
    journal_tail_record_hash: str | None
    journal_lineage_id: UUID
    anchor_hash: str = Field(min_length=64, max_length=64)

    @field_validator(
        "snapshot_sequence", "journal_processing_high_water", mode="before"
    )
    @classmethod
    def require_exact_persisted_sequence(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("persisted processing sequence must be an integer")
        return value


class StateWALInspection(_StrictModel):
    exists: bool
    active_manifest: Manifest | None
    records: tuple[BaselineRecord | TransitionRecord, ...]
    record_hashes: tuple[str, ...]
    baseline_record_id: UUID | None
    baseline_record_hash: str | None
    latest_snapshot_sequence: int | None
    latest_snapshot_hash: str | None


class StateWALFieldChange(_StrictModel):
    field: str
    before: Any
    after: Any


class DryRunDiff(_StrictModel):
    target_snapshot_sequence: int
    target_snapshot_hash: str
    current_snapshot_sequence: int
    current_snapshot_hash: str
    changes: tuple[StateWALFieldChange, ...]
    records_replayed: int
    external_side_effects_replayed: Literal[False]


_FORBIDDEN = frozenset(
    {
        "hiddenthought",
        "privatereasoning",
        "reasoning",
        "chainofthought",
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
        "attachments",
        "eventpayload",
        "requestpayload",
        "debugtrace",
        "transcript",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _reject_private(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if normalize_private_key(str(key)) in _FORBIDDEN:
                raise StateWALFormatError("WAL data violates privacy policy")
            _reject_private(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private(child)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _domain_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode() + b"\0" + _canonical(value)).hexdigest()


def _record_json(model: BaseModel, domain: str) -> tuple[dict[str, Any], str]:
    raw = model.model_dump(mode="json")
    unsigned = {key: value for key, value in raw.items() if key != "record_hash"}
    digest = _domain_hash(domain, unsigned)
    raw["record_hash"] = digest
    return raw, digest


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_sequence(value: object, field: str) -> int:
    if (
        type(value) is not int
        or value < 0
        or value > MAX_PERSISTED_EVENT_SEQUENCE
    ):
        raise StateWALFormatError(f"{field} is outside its bound")
    return value


def _snapshot_sequence(snapshot: CompatibleAgentStateSnapshot) -> int:
    try:
        value = snapshot.last_processed_event_sequence
    except AttributeError:
        raise StateWALFormatError("snapshot sequence is unavailable") from None
    return _validate_sequence(value, "snapshot sequence")


@dataclass(slots=True)
class _GenerationScan:
    """Verifier state carried from one verified WAL line to the next."""

    records: list[BaselineRecord | TransitionRecord] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)
    ids: set[UUID] = field(default_factory=set)
    events: set[UUID] = field(default_factory=set)
    state_sequence: int | None = None
    state_hash: str | None = None
    processing_sequence: int | None = None

    def copy(self) -> _GenerationScan:
        return _GenerationScan(
            records=list(self.records),
            hashes=list(self.hashes),
            ids=set(self.ids),
            events=set(self.events),
            state_sequence=self.state_sequence,
            state_hash=self.state_hash,
            processing_sequence=self.processing_sequence,
        )


@dataclass(frozen=True, slots=True)
class _VerifiedGeneration:
    """In-process record of a fully verified generation prefix (not durable)."""

    manifest_hash: str
    generation_id: UUID
    device: int
    inode: int
    length: int
    last_line: bytes
    scan: _GenerationScan


class StateWAL:
    """A durable WAL that has no authority to publish canonical state."""

    def __init__(
        self,
        directory: str | Path,
        *,
        canonical_bytes: Callable[[CompatibleAgentStateSnapshot], bytes] | None = None,
        snapshot_hash: Callable[[CompatibleAgentStateSnapshot], str] | None = None,
        failure_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(directory)
        self._canonical_bytes = canonical_bytes or AgentStateStore._canonical_bytes
        self._snapshot_hash = snapshot_hash
        self._failure_hook = failure_hook
        self._lock = RLock()
        self._verified: _VerifiedGeneration | None = None

    def _stage(self, stage: str) -> None:
        if self._failure_hook is not None:
            self._failure_hook(stage)

    @staticmethod
    def _check_dir(path: Path, *, create: bool) -> None:
        descriptor = -1
        try:
            descriptor = (
                StateWAL._open_or_create_private_dir(path)
                if create
                else StateWAL._open_private_dir(path, require_private=False)
            )
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
                raise OSError
            if stat.S_IMODE(status.st_mode) != 0o700:
                os.fchmod(descriptor, 0o700)
                os.fsync(descriptor)
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
                    raise OSError
        except Exception:
            raise StateWALPermissionError("WAL directory is unsafe") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _validate_descriptor(descriptor: int, *, regular: bool = True) -> None:
        status = os.fstat(descriptor)
        if (
            (regular and not stat.S_ISREG(status.st_mode))
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o600
        ):
            raise OSError

    @staticmethod
    def _open_private_dir(path: Path, *, require_private: bool = True) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        absolute = path.absolute()
        descriptor = os.open(absolute.anchor, flags)
        try:
            for component in absolute.parts[1:]:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or (require_private and stat.S_IMODE(status.st_mode) != 0o700)
            ):
                raise OSError
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_or_create_private_dir(path: Path) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        absolute = path.absolute()
        descriptor = os.open(absolute.anchor, flags)
        try:
            for component in absolute.parts[1:]:
                try:
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _prepare_for_write(self) -> None:
        self._check_dir(self.root.parent, create=True)
        self._check_dir(self.root, create=True)
        self._check_dir(self.root / "generations", create=True)

    def _prepare_for_read(self) -> bool:
        try:
            self._check_dir(self.root.parent, create=False)
            self._check_dir(self.root, create=False)
            self._check_dir(self.root / "generations", create=False)
        except StateWALPermissionError:
            if not self.root.exists():
                return False
            raise
        return True

    @staticmethod
    def _read_regular(path: Path) -> bytes:
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = StateWAL._open_private_dir(path.parent)
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            StateWAL._validate_descriptor(descriptor)
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                return source.read()
        except FileNotFoundError:
            raise StateWALMissing("WAL artifact is absent") from None
        except Exception:
            raise StateWALFormatError("WAL artifact is invalid") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        descriptor = StateWAL._open_private_dir(path, require_private=False)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(path.parent)
            try:
                existing = os.stat(
                    path.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    not stat.S_ISREG(existing.st_mode)
                    or existing.st_uid != os.geteuid()
                    or stat.S_IMODE(existing.st_mode) != 0o600
                ):
                    raise OSError
            self._stage("temp_write")
            descriptor = os.open(
                temporary.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                output.write(_canonical(value))
                output.flush()
                self._stage("temp_fsync")
                os.fsync(output.fileno())
            self._stage("atomic_replace")
            os.replace(
                temporary.name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            self._stage("parent_fsync")
            os.fsync(parent_descriptor)
        except Exception:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise StateWALError("WAL durable write failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _snapshot_hash_value(self, snapshot: CompatibleAgentStateSnapshot) -> str:
        try:
            _snapshot_sequence(snapshot)
            # Validate the union directly.  Round-tripping through a generic
            # dump is needlessly lossy for retained v1/v2 records (and makes it
            # easier for a newer adapter to normalize a retained shape).
            # Keeping the discriminated model intact is what preserves both
            # legacy snapshot bytes and their hashes in mixed generations.
            validated = validate_compatible_agent_state_snapshot(snapshot)
            # The v1/v2 representations are part of the retained WAL contract.
            # Do not route them through a newer canonicalizer: doing so changes
            # old snapshot identities and every record hash that embeds them.
            payload = self._canonical_bytes(validated)
            return (
                self._snapshot_hash(validated)
                if self._snapshot_hash
                else hashlib.sha256(payload).hexdigest()
            )
        except Exception:
            raise StateWALFormatError("snapshot is invalid") from None

    def _generation_path(self, generation_id: UUID) -> Path:
        return self.root / "generations" / f"{generation_id}.jsonl"

    def _begin_generation(
        self,
        snapshot: CompatibleAgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        reason: RecoveryReason = RecoveryReason.BOOTSTRAP,
        generation_id: UUID | None = None,
        predecessor_generation_id: UUID | None = None,
        predecessor_generation_hash: str | None = None,
        external_reconciliation_required: bool = False,
        prepared_recovery_id: UUID | None = None,
        allow_invalid_current: bool = False,
    ) -> Manifest:
        _snapshot_sequence(snapshot)
        _validate_sequence(
            journal_processing_high_water, "journal processing high-water"
        )
        self._prepare_for_write()
        invalid_current_detected = False
        try:
            current = self.inspect_optional()
        except StateWALError:
            if prepared_recovery_id is None or not (
                external_reconciliation_required or allow_invalid_current
            ):
                raise
            current = None
            invalid_current_detected = True
        if (
            current is not None
            and current.active_manifest is not None
            and current.active_manifest.external_reconciliation_required
            and not external_reconciliation_required
        ):
            raise StateWALConflictError(
                "external reconciliation gate cannot be cleared by StateWAL"
            )
        generation = generation_id or uuid4()
        path = self._generation_path(generation)
        if path.exists():
            try:
                inspection = self.inspect()
            except StateWALError:
                inspection = None
            if (
                inspection is not None
                and inspection.active_manifest
                and inspection.active_manifest.active_generation_id == generation
                and inspection.latest_snapshot_sequence
                == snapshot.last_processed_event_sequence
                and inspection.latest_snapshot_hash
                == self._snapshot_hash_value(snapshot)
            ):
                return inspection.active_manifest
            try:
                baseline, baseline_hash = self._inspect_orphan_baseline(generation)
            except StateWALConflictError:
                if prepared_recovery_id is None:
                    raise
                self._preserve_partial_generation(path, prepared_recovery_id)
            else:
                if (
                    baseline.baseline_snapshot != snapshot
                    or baseline.journal_processing_high_water
                    != journal_processing_high_water
                    or baseline.reason is not reason
                    or baseline.predecessor_generation_id != predecessor_generation_id
                    or baseline.predecessor_generation_hash
                    != predecessor_generation_hash
                ):
                    raise StateWALConflictError("generation already exists")
                return self._publish_manifest(
                    baseline, baseline_hash, external_reconciliation_required
                )
        snapshot_hash = self._snapshot_hash_value(snapshot)
        baseline = BaselineRecord(
            record_type="baseline",
            schema_version=1,
            command_version=1,
            record_id=uuid4(),
            generation_id=generation,
            created_at=_now(),
            baseline_snapshot_sequence=snapshot.last_processed_event_sequence,
            baseline_snapshot_hash=snapshot_hash,
            baseline_snapshot=snapshot,
            journal_processing_high_water=journal_processing_high_water,
            predecessor_generation_id=predecessor_generation_id,
            predecessor_generation_hash=predecessor_generation_hash,
            reason=reason,
            previous_record_hash=None,
            record_hash="0" * 64,
        )
        raw, baseline_hash = _record_json(baseline, "suzka.state-wal.baseline")
        self._write_new_generation(path, raw)
        if invalid_current_detected:
            assert prepared_recovery_id is not None
            self._preserve_invalid_manifest(prepared_recovery_id)
        return self._publish_manifest(
            baseline, baseline_hash, external_reconciliation_required
        )

    def _write_new_generation(self, path: Path, raw: dict[str, Any]) -> None:
        """Create a generation file holding exactly one baseline, durably."""
        parent_descriptor = -1
        descriptor = -1
        try:
            self._stage("generation_write")
            parent_descriptor = self._open_private_dir(path.parent)
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                self._validate_descriptor(output.fileno())
                output.write(_canonical(raw))
                output.flush()
                self._stage("generation_fsync")
                os.fsync(output.fileno())
            os.fsync(parent_descriptor)
        except Exception:
            raise StateWALError("WAL generation write failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    # --- ordinary generation rollover (#382 A, IP-3b) --------------------------

    def prepare_rollover_generation(
        self,
        *,
        generation_id: UUID,
        journal_processing_high_water: int,
    ) -> tuple[BaselineRecord, str]:
        """Write, but do not publish, a new generation for the same state.

        The baseline is the latest verified snapshot of the active generation
        and is chained to that generation's last record hash. The manifest is
        unchanged until ``publish_rollover_generation``.
        """
        _validate_sequence(
            journal_processing_high_water, "journal processing high-water"
        )
        with self._lock:
            self._prepare_for_write()
            # The cached inspection is sufficient: every line of the active
            # generation passed the same verification when it was appended,
            # and in-place tampering of verified bytes is outside the
            # single-cooperative-writer threat model (IP-1, #384); it is caught
            # by the full verification at process start. A full re-verification
            # here measured ~0.37 s per record (~12 s per 32-transition
            # rollover) on the rollover turn, for no gain inside that model.
            current = self.inspect()
            manifest = current.active_manifest
            if manifest is None or not current.records:
                raise StateWALIntegrityError("rollover requires an active generation")
            if manifest.external_reconciliation_required:
                raise StateWALConflictError("rollover is forbidden while gated")
            if generation_id == manifest.active_generation_id:
                raise StateWALConflictError("rollover must change generation")
            latest = current.records[-1]
            snapshot = (
                latest.baseline_snapshot
                if isinstance(latest, BaselineRecord)
                else latest.candidate_snapshot
            )
            assert current.latest_snapshot_hash is not None
            baseline = BaselineRecord(
                record_type="baseline",
                schema_version=1,
                command_version=1,
                record_id=uuid4(),
                generation_id=generation_id,
                created_at=_now(),
                baseline_snapshot_sequence=snapshot.last_processed_event_sequence,
                baseline_snapshot_hash=current.latest_snapshot_hash,
                baseline_snapshot=snapshot,
                journal_processing_high_water=journal_processing_high_water,
                predecessor_generation_id=manifest.active_generation_id,
                predecessor_generation_hash=current.record_hashes[-1],
                reason=RecoveryReason.ROLLOVER,
                previous_record_hash=None,
                record_hash="0" * 64,
            )
            raw, baseline_hash = _record_json(baseline, "suzka.state-wal.baseline")
            self._write_new_generation(self._generation_path(generation_id), raw)
            return BaselineRecord.model_validate_json(_canonical(raw)), baseline_hash

    def publish_rollover_generation(
        self,
        *,
        generation_id: UUID,
        baseline_record_hash: str,
        journal_processing_high_water: int,
    ) -> Manifest:
        """Atomically make a prepared rollover generation active."""
        with self._lock:
            self._prepare_for_write()
            current = self.inspect()
            manifest = current.active_manifest
            if manifest is None or not current.records:
                raise StateWALIntegrityError("rollover requires an active generation")
            if manifest.external_reconciliation_required:
                raise StateWALConflictError("rollover is forbidden while gated")
            baseline, baseline_hash = self._inspect_orphan_baseline(generation_id)
            if (
                baseline_hash != baseline_record_hash
                or baseline.reason is not RecoveryReason.ROLLOVER
                or baseline.predecessor_generation_id != manifest.active_generation_id
                or baseline.predecessor_generation_hash != current.record_hashes[-1]
                or baseline.baseline_snapshot_sequence
                != current.latest_snapshot_sequence
                or baseline.baseline_snapshot_hash != current.latest_snapshot_hash
                or baseline.journal_processing_high_water
                != journal_processing_high_water
            ):
                raise StateWALConflictError("rollover generation does not chain")
            self._stage("rollover_publish")
            return self._publish_manifest(baseline, baseline_hash, False)

    def inspect_rollover_generation(
        self, generation_id: UUID
    ) -> tuple[BaselineRecord, str] | None:
        """The unpublished-or-published single baseline of a generation, if valid."""
        with self._lock:
            try:
                return self._inspect_orphan_baseline(generation_id)
            except StateWALConflictError:
                return None

    def read_manifest_optional(self) -> Manifest | None:
        """The hash-verified manifest alone, without reading any generation.

        ``None`` when there is no WAL or no manifest. Used where a decision
        depends only on which generation is published, even if that
        generation itself is unreadable.
        """
        with self._lock:
            if not self._prepare_for_read():
                return None
            try:
                manifest_raw = json.loads(
                    self._read_regular(self.root / "manifest.json")
                )
            except StateWALMissing:
                return None
            except StateWALError:
                raise
            except Exception:
                raise StateWALIntegrityError("WAL manifest is invalid") from None
            try:
                manifest = Manifest.model_validate_json(_canonical(manifest_raw))
                expected = _domain_hash(
                    "suzka.state-wal.manifest",
                    {
                        key: value
                        for key, value in manifest_raw.items()
                        if key != "manifest_hash"
                    },
                )
                if manifest.manifest_hash != expected:
                    raise ValueError
            except Exception:
                raise StateWALIntegrityError("WAL manifest is invalid") from None
            return manifest

    def quarantine_generation(self, generation_id: UUID, reason_id: UUID) -> bool:
        """Move a never-published generation file aside (#382 D1 P2-4)."""
        with self._lock:
            path = self._generation_path(generation_id)
            if not path.exists():
                return False
            manifest = self.read_manifest_optional()
            if manifest is not None and manifest.active_generation_id == generation_id:
                raise StateWALConflictError("active generation cannot be quarantined")
            self._preserve_partial_generation(path, reason_id)
            return True

    def prune_generations(self) -> tuple[UUID, ...]:
        """Delete generations other than active, its predecessor and boot anchor.

        Retains at most three generations (#382 D1 P0, Human decision). The
        boot anchor is re-read under the lock immediately before unlinking.
        Unparseable names and preserved ``.invalid`` files are left alone.
        """
        with self._lock:
            self._prepare_for_write()
            manifest = self.inspect().active_manifest
            if manifest is None:
                raise StateWALIntegrityError("pruning requires an active generation")
            keep = {manifest.active_generation_id}
            if manifest.predecessor_generation_id is not None:
                keep.add(manifest.predecessor_generation_id)
            anchor = self.inspect_boot_anchor_optional()
            if anchor is not None:
                keep.add(anchor.generation_id)
            removed: list[UUID] = []
            directory = self.root / "generations"
            parent_descriptor = -1
            try:
                parent_descriptor = self._open_private_dir(directory)
                for name in sorted(os.listdir(parent_descriptor)):
                    if not name.endswith(".jsonl"):
                        continue
                    try:
                        generation = UUID(name[: -len(".jsonl")])
                    except ValueError:
                        continue
                    if str(generation) != name[: -len(".jsonl")] or generation in keep:
                        continue
                    self._stage("generation_prune")
                    os.unlink(name, dir_fd=parent_descriptor)
                    removed.append(generation)
                if removed:
                    os.fsync(parent_descriptor)
            except StateWALError:
                raise
            except Exception:
                raise StateWALError("WAL generation pruning failed") from None
            finally:
                if parent_descriptor >= 0:
                    os.close(parent_descriptor)
            return tuple(removed)

    def generation_size(self) -> int:
        """Bytes of the active generation file."""
        with self._lock:
            manifest = self.inspect().active_manifest
            if manifest is None:
                raise StateWALIntegrityError("no active generation")
            return self._generation_path(manifest.active_generation_id).stat().st_size

    def _preserve_partial_generation(self, path: Path, recovery_id: UUID) -> None:
        preserved = path.with_name(f".{path.name}.{recovery_id}.{uuid4()}.invalid")
        parent_descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(path.parent)
            os.replace(
                path.name,
                preserved.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except Exception:
            raise StateWALError("partial WAL generation cannot be preserved") from None
        finally:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _preserve_invalid_manifest(self, recovery_id: UUID) -> None:
        path = self.root / "manifest.json"
        preserved = self.root / f".manifest.json.{recovery_id}.{uuid4()}.invalid"
        parent_descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(self.root)
            os.replace(
                path.name,
                preserved.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except FileNotFoundError:
            return
        except Exception:
            raise StateWALError("invalid WAL manifest cannot be preserved") from None
        finally:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _publish_manifest(
        self,
        baseline: BaselineRecord,
        baseline_hash: str,
        external_reconciliation_required: bool,
    ) -> Manifest:
        manifest_base = Manifest(
            schema_version=1,
            command_version=1,
            active_generation_id=baseline.generation_id,
            active_baseline_record_id=baseline.record_id,
            active_baseline_record_hash=baseline_hash,
            predecessor_generation_id=baseline.predecessor_generation_id,
            predecessor_generation_hash=baseline.predecessor_generation_hash,
            external_reconciliation_required=external_reconciliation_required,
            manifest_hash="0" * 64,
        )
        manifest_raw = manifest_base.model_dump(mode="json")
        manifest_raw["manifest_hash"] = _domain_hash(
            "suzka.state-wal.manifest",
            {
                key: value
                for key, value in manifest_raw.items()
                if key != "manifest_hash"
            },
        )
        self._atomic_write(self.root / "manifest.json", manifest_raw)
        return Manifest.model_validate_json(_canonical(manifest_raw))

    def _inspect_orphan_baseline(
        self, generation_id: UUID
    ) -> tuple[BaselineRecord, str]:
        try:
            lines = self._read_regular(self._generation_path(generation_id)).splitlines(
                keepends=True
            )
            if len(lines) != 1:
                raise ValueError
            if not lines[0].endswith(b"\n"):
                raise ValueError
            preflight_attention_json(lines[0])
            raw = json.loads(lines[0])
            baseline = BaselineRecord.model_validate_json(_canonical(raw))
            record_hash = _record_json(baseline, "suzka.state-wal.baseline")[1]
            if (
                baseline.generation_id != generation_id
                or baseline.record_hash != record_hash
                or baseline.baseline_snapshot_sequence
                != baseline.baseline_snapshot.last_processed_event_sequence
                or baseline.baseline_snapshot_hash
                != self._snapshot_hash_value(baseline.baseline_snapshot)
            ):
                raise ValueError
            return baseline, record_hash
        except RecursionError:
            raise StateWALConflictError(
                "existing generation cannot be resumed"
            ) from None
        except Exception:
            raise StateWALConflictError(
                "existing generation cannot be resumed"
            ) from None

    def begin_generation(
        self,
        snapshot: CompatibleAgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        reason: RecoveryReason = RecoveryReason.BOOTSTRAP,
        generation_id: UUID | None = None,
        predecessor_generation_id: UUID | None = None,
        predecessor_generation_hash: str | None = None,
        external_reconciliation_required: bool = False,
    ) -> Manifest:
        with self._lock:
            return self._begin_generation(
                snapshot,
                journal_processing_high_water,
                reason=reason,
                generation_id=generation_id,
                predecessor_generation_id=predecessor_generation_id,
                predecessor_generation_hash=predecessor_generation_hash,
                external_reconciliation_required=external_reconciliation_required,
            )

    def resume_prepared_generation(
        self,
        snapshot: CompatibleAgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        recovery_id: UUID,
        reason: RecoveryReason,
        generation_id: UUID,
        predecessor_generation_id: UUID | None,
        predecessor_generation_hash: str | None,
        external_reconciliation_required: bool,
    ) -> Manifest:
        """Resume only a Journal-prepared recovery while preserving partial bytes."""

        with self._lock:
            return self._begin_generation(
                snapshot,
                journal_processing_high_water,
                reason=reason,
                generation_id=generation_id,
                predecessor_generation_id=predecessor_generation_id,
                predecessor_generation_hash=predecessor_generation_hash,
                external_reconciliation_required=external_reconciliation_required,
                prepared_recovery_id=recovery_id,
            )

    def rebaseline_prepared_current(
        self,
        snapshot: CompatibleAgentStateSnapshot,
        journal_processing_high_water: int,
        *,
        recovery_id: UUID,
        generation_id: UUID,
        external_reconciliation_required: bool = False,
    ) -> Manifest:
        """Replace invalid active WAL metadata from Journal-confirmed current state."""

        with self._lock:
            return self._begin_generation(
                snapshot,
                journal_processing_high_water,
                reason=RecoveryReason.EXACT_CURRENT_REPAIR,
                generation_id=generation_id,
                predecessor_generation_id=None,
                predecessor_generation_hash=None,
                external_reconciliation_required=external_reconciliation_required,
                prepared_recovery_id=recovery_id,
                allow_invalid_current=True,
            )

    def replace_unanchored_provisional(
        self,
        snapshot: CompatibleAgentStateSnapshot,
        journal_processing_high_water: int,
    ) -> Manifest:
        """Replace WAL bytes proven not to be anchored by a v2 Journal."""

        with self._lock:
            try:
                current = self.inspect_optional()
            except StateWALError:
                current = None
            if current is not None and current.exists:
                raise StateWALConflictError("provisional WAL is still valid")
            if self.inspect_boot_anchor_optional() is not None:
                raise StateWALConflictError("provisional WAL has a boot anchor")
            return self._begin_generation(
                snapshot,
                journal_processing_high_water,
                reason=RecoveryReason.BOOTSTRAP,
                generation_id=uuid4(),
                predecessor_generation_id=None,
                predecessor_generation_hash=None,
                external_reconciliation_required=False,
                prepared_recovery_id=uuid4(),
                allow_invalid_current=True,
            )

    bootstrap = begin_generation

    def exists(self) -> bool:
        try:
            if not self._prepare_for_read():
                return False
            status = (self.root / "manifest.json").lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o600
            ):
                raise StateWALPermissionError("WAL manifest is unsafe")
            return True
        except FileNotFoundError:
            return False

    def inspect_optional(self) -> StateWALInspection:
        return self._inspect_optional(full_verification=False)

    def _inspect_full(self) -> StateWALInspection:
        """Inspect with a full verification that bypasses the in-process cache.

        Recovery-facing reads (``reconstruct``) use this so that they never
        depend on the cooperative-writer assumption of the cached path.
        """
        inspection = self._inspect_optional(full_verification=True)
        if not inspection.exists:
            raise StateWALMissing("WAL is absent")
        return inspection

    def _inspect_optional(self, *, full_verification: bool) -> StateWALInspection:
        if not self._prepare_for_read():
            return StateWALInspection(
                exists=False,
                active_manifest=None,
                records=(),
                record_hashes=(),
                baseline_record_id=None,
                baseline_record_hash=None,
                latest_snapshot_sequence=None,
                latest_snapshot_hash=None,
            )
        try:
            manifest_raw = json.loads(self._read_regular(self.root / "manifest.json"))
        except StateWALMissing:
            return StateWALInspection(
                exists=False,
                active_manifest=None,
                records=(),
                record_hashes=(),
                baseline_record_id=None,
                baseline_record_hash=None,
                latest_snapshot_sequence=None,
                latest_snapshot_hash=None,
            )
        except StateWALError:
            raise
        except Exception:
            raise StateWALIntegrityError("WAL manifest is invalid") from None
        try:
            manifest = Manifest.model_validate_json(_canonical(manifest_raw))
            expected = _domain_hash(
                "suzka.state-wal.manifest",
                {
                    key: value
                    for key, value in manifest_raw.items()
                    if key != "manifest_hash"
                },
            )
            if manifest.manifest_hash != expected:
                raise ValueError
            scan = (
                self._read_generation(manifest)
                if full_verification
                else self._read_generation_cached(manifest)
            )
        except StateWALMissing:
            raise StateWALIntegrityError("active WAL generation is absent") from None
        except StateWALError:
            raise
        except Exception:
            raise StateWALIntegrityError("WAL manifest is invalid") from None
        records = scan.records
        hashes = scan.hashes
        latest = records[-1]
        snapshot = (
            latest.baseline_snapshot
            if isinstance(latest, BaselineRecord)
            else latest.candidate_snapshot
        )
        # ``scan.state_hash`` is the verified snapshot hash of ``latest``: the
        # line verifier recomputed it from the snapshot before accepting the
        # record, so it need not be recomputed for every inspection.
        assert scan.state_hash is not None
        return StateWALInspection(
            exists=True,
            active_manifest=manifest,
            records=tuple(records),
            record_hashes=tuple(hashes),
            baseline_record_id=records[0].record_id,
            baseline_record_hash=hashes[0],
            latest_snapshot_sequence=snapshot.last_processed_event_sequence,
            latest_snapshot_hash=scan.state_hash,
        )

    def inspect(self) -> StateWALInspection:
        inspection = self.inspect_optional()
        if not inspection.exists:
            raise StateWALMissing("WAL is absent")
        return inspection

    def clear_external_reconciliation_gate(self, expected: Manifest) -> Manifest:
        """Clear the gate with a manifest-identity compare-and-swap.

        This is deliberately separate from generation creation: a recovery
        coordinator may only clear the exact manifest it inspected and proved.
        """
        with self._lock:
            expected_raw = expected.model_dump(mode="json")
            expected_hash = _domain_hash(
                "suzka.state-wal.manifest",
                {key: value for key, value in expected_raw.items()
                 if key != "manifest_hash"},
            )
            if expected.manifest_hash != expected_hash:
                raise StateWALGateClearProofError("WAL manifest proof is invalid")
            current = self.inspect()
            manifest = current.active_manifest
            if manifest is None:
                raise StateWALGateClearProofError("WAL manifest is absent")
            if manifest.external_reconciliation_required:
                if manifest != expected:
                    raise StateWALGateClearProofError("WAL manifest proof is stale")
                clear_base = manifest.model_copy(
                    update={"external_reconciliation_required": False,
                            "manifest_hash": "0" * 64}
                )
            else:
                # A crash after the clear-prepared Journal record can leave
                # this already applied.  Re-verify, but never rewrite it.
                if manifest.active_generation_id != expected.active_generation_id:
                    raise StateWALGateClearProofError("WAL generation proof is stale")
                if (
                    manifest.active_baseline_record_id != expected.active_baseline_record_id
                    or manifest.active_baseline_record_hash != expected.active_baseline_record_hash
                    or manifest.predecessor_generation_id != expected.predecessor_generation_id
                    or manifest.predecessor_generation_hash != expected.predecessor_generation_hash
                ):
                    raise StateWALGateClearProofError("WAL manifest identity is stale")
                return manifest
            raw = clear_base.model_dump(mode="json")
            raw["manifest_hash"] = _domain_hash(
                "suzka.state-wal.manifest",
                {key: value for key, value in raw.items() if key != "manifest_hash"},
            )
            self._atomic_write(self.root / "manifest.json", raw)
            return Manifest.model_validate_json(_canonical(raw))

    clear_gate_if_matches = clear_external_reconciliation_gate

    @staticmethod
    def _read_generation_bytes(
        path: Path, cached: _VerifiedGeneration | None
    ) -> tuple[bytes, bool, int, int]:
        """Read a generation like ``_read_regular``.

        When ``cached`` describes the same file (dev/inode), the file has not
        shrunk, and the last verified line is still byte-identical at its
        verified position, only the bytes after the verified prefix are read
        and ``(suffix, True, dev, inode)`` is returned.  Otherwise the whole
        file is read and ``(content, False, dev, inode)`` is returned.
        """
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = StateWAL._open_private_dir(path.parent)
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            StateWAL._validate_descriptor(descriptor)
            status = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                if (
                    cached is not None
                    and cached.device == status.st_dev
                    and cached.inode == status.st_ino
                    and status.st_size >= cached.length
                ):
                    tail_start = cached.length - len(cached.last_line)
                    source.seek(tail_start)
                    data = source.read()
                    if data.startswith(cached.last_line):
                        return (
                            data[len(cached.last_line) :],
                            True,
                            status.st_dev,
                            status.st_ino,
                        )
                    source.seek(0)
                return source.read(), False, status.st_dev, status.st_ino
        except FileNotFoundError:
            raise StateWALMissing("WAL artifact is absent") from None
        except Exception:
            raise StateWALFormatError("WAL artifact is invalid") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def invalidate_verification_cache(self) -> None:
        """Forget the in-process verified prefix; the next read verifies all."""
        with self._lock:
            self._verified = None

    def _read_generation_cached(self, manifest: Manifest) -> _GenerationScan:
        """Verify the active generation, re-verifying only appended lines.

        The cache is an in-process optimisation, never an authority.  It is
        used only for the same manifest and the same file (dev/inode) that has
        not shrunk and whose last verified line is still byte-identical at
        its verified position; then only appended lines are verified.
        Otherwise, or after any failure, the whole generation is verified
        again.  A fresh instance (every process start) has no cache, so
        startup always performs a full verification.

        Threat model: the WAL has a single cooperative writer (this process,
        under the Journal lease).  An in-place rewrite of an earlier,
        already-verified line that preserves the file size and the last line
        is not detected by the cached path; it is detected by the next full
        verification (process start, ``invalidate_verification_cache`` or the
        recovery-facing ``reconstruct``, which always verifies fully).

        Costs and sharing: the cache retains the verified records of the
        active generation for the process lifetime, so memory is proportional
        to the generation size (bounded once generations roll over, #382).
        Returned inspections share the same frozen record objects; callers
        must not mutate nested snapshot containers.  The whole read-verify
        step runs under ``self._lock`` and is therefore serialised with
        appends.
        """
        with self._lock:
            cached = self._verified
            if cached is not None and (
                cached.manifest_hash != manifest.manifest_hash
                or cached.generation_id != manifest.active_generation_id
            ):
                cached = None
            data, incremental, device, inode = self._read_generation_bytes(
                self._generation_path(manifest.active_generation_id), cached
            )
            scan: _GenerationScan
            if incremental:
                assert cached is not None
                scan = cached.scan.copy()
                base_length = cached.length
                previous_last_line = cached.last_line
            else:
                scan = _GenerationScan()
                base_length = 0
                previous_last_line = b""
            self._verified = None
            lines = data.splitlines(keepends=True)
            self._verify_generation_lines(manifest, lines, scan)
            if (
                not scan.records
                or scan.records[0].record_id != manifest.active_baseline_record_id
                or scan.hashes[0] != manifest.active_baseline_record_hash
            ):
                raise StateWALIntegrityError("WAL manifest does not anchor generation")
            self._verified = _VerifiedGeneration(
                manifest_hash=manifest.manifest_hash,
                generation_id=manifest.active_generation_id,
                device=device,
                inode=inode,
                length=base_length + len(data),
                last_line=lines[-1] if lines else previous_last_line,
                scan=scan.copy(),
            )
            return scan

    def _read_generation(self, manifest: Manifest) -> _GenerationScan:
        """Fully verify the active generation without using or updating the cache."""
        raw_lines = self._read_regular(
            self._generation_path(manifest.active_generation_id)
        ).splitlines(keepends=True)
        scan = _GenerationScan()
        self._verify_generation_lines(manifest, raw_lines, scan)
        if (
            not scan.records
            or scan.records[0].record_id != manifest.active_baseline_record_id
            or scan.hashes[0] != manifest.active_baseline_record_hash
        ):
            raise StateWALIntegrityError("WAL manifest does not anchor generation")
        return scan

    def _verify_generation_lines(
        self,
        manifest: Manifest,
        raw_lines: list[bytes],
        scan: _GenerationScan,
    ) -> None:
        records = scan.records
        hashes = scan.hashes
        ids = scan.ids
        events = scan.events
        state_sequence = scan.state_sequence
        state_hash = scan.state_hash
        processing_sequence = scan.processing_sequence
        for line in raw_lines:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError
                preflight_attention_json(line)
                raw = json.loads(line)
                kind = raw.get("record_type")
                model = (
                    BaselineRecord
                    if kind == "baseline"
                    else TransitionRecord
                    if kind == "transition"
                    else None
                )
                if model is None:
                    raise ValueError
                record = model.model_validate_json(_canonical(raw))
                record_hash = _record_json(record, f"suzka.state-wal.{kind}")[1]
                if (
                    not _HASH_RE.fullmatch(record.record_hash)
                    or record.record_hash != record_hash
                ):
                    raise ValueError
                if record.record_id in ids or (
                    isinstance(record, TransitionRecord) and record.event_id in events
                ):
                    raise ValueError
                if isinstance(record, TransitionRecord):
                    if (
                        state_sequence is None
                        or state_hash is None
                        or processing_sequence is None
                        or record.previous_record_hash != hashes[-1]
                        or record.prior_snapshot_sequence != state_sequence
                        or record.prior_snapshot_hash != state_hash
                        or record.processing_sequence <= processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.candidate_snapshot.last_processed_event_sequence
                        or record.candidate_snapshot_hash
                        != self._snapshot_hash_value(record.candidate_snapshot)
                    ):
                        raise ValueError
                    if record.candidate_snapshot_sequence <= state_sequence:
                        raise ValueError
                else:
                    if records or record.previous_record_hash is not None:
                        raise ValueError
                    if (
                        record.baseline_snapshot_sequence
                        != record.baseline_snapshot.last_processed_event_sequence
                        or record.baseline_snapshot_hash
                        != self._snapshot_hash_value(record.baseline_snapshot)
                        or record.journal_processing_high_water
                        < record.baseline_snapshot_sequence
                    ):
                        raise ValueError
                if isinstance(record, TransitionRecord):
                    processing_sequence = record.processing_sequence
                    state_sequence = record.candidate_snapshot_sequence
                    state_hash = record.candidate_snapshot_hash
                else:
                    state_sequence = record.baseline_snapshot_sequence
                    state_hash = record.baseline_snapshot_hash
                    processing_sequence = record.journal_processing_high_water
                if record.generation_id != manifest.active_generation_id:
                    raise ValueError
                # ``scan`` is mutated only after the line passed every check.
                ids.add(record.record_id)
                if isinstance(record, TransitionRecord):
                    events.add(record.event_id)
                records.append(record)
                hashes.append(record.record_hash)
                scan.state_sequence = state_sequence
                scan.state_hash = state_hash
                scan.processing_sequence = processing_sequence
            except RecursionError:
                raise StateWALIntegrityError("WAL generation is invalid") from None
            except Exception:
                raise StateWALIntegrityError("WAL generation is invalid") from None

    def _append_transition(
        self,
        *,
        event_id: UUID,
        event_type: str,
        event_source: str,
        processing_sequence: int,
        prior_snapshot: CompatibleAgentStateSnapshot,
        candidate_snapshot: CompatibleAgentStateSnapshot,
    ) -> TransitionRecord:
        _validate_sequence(processing_sequence, "processing sequence")
        _snapshot_sequence(prior_snapshot)
        _snapshot_sequence(candidate_snapshot)
        inspection = self.inspect()
        assert inspection.active_manifest is not None
        if any(
            isinstance(record, TransitionRecord) and record.event_id == event_id
            for record in inspection.records
        ):
            raise StateWALConflictError("event identifier is already retained in WAL")
        prior_hash = self._snapshot_hash_value(prior_snapshot)
        candidate_hash = self._snapshot_hash_value(candidate_snapshot)
        # Ordinary transitions can advance the current authority schema but
        # cannot downgrade any v8+ authority. Explicit true rollback uses a
        # new generation instead.
        if (
            prior_snapshot.schema_version >= 8
            and candidate_snapshot.schema_version < prior_snapshot.schema_version
        ):
            raise StateWALConflictError(
                "ordinary v8-or-newer state transition cannot downgrade schema"
            )
        if (
            inspection.latest_snapshot_sequence
            != prior_snapshot.last_processed_event_sequence
            or inspection.latest_snapshot_hash != prior_hash
        ):
            raise StateWALConflictError("state lineage does not match WAL")
        if (
            candidate_snapshot.last_processed_event_sequence
            <= prior_snapshot.last_processed_event_sequence
        ):
            raise StateWALConflictError("state transition sequence must increase")
        if candidate_snapshot.last_processed_event_sequence != processing_sequence:
            raise StateWALConflictError(
                "candidate sequence must match processing sequence"
            )
        record = TransitionRecord(
            record_type="transition",
            schema_version=1,
            command_version=1,
            record_id=uuid4(),
            generation_id=inspection.active_manifest.active_generation_id,
            created_at=_now(),
            event_id=event_id,
            event_type=event_type,
            event_source=event_source,
            processing_sequence=processing_sequence,
            prior_snapshot_sequence=prior_snapshot.last_processed_event_sequence,
            prior_snapshot_hash=prior_hash,
            candidate_snapshot_sequence=candidate_snapshot.last_processed_event_sequence,
            candidate_snapshot_hash=candidate_hash,
            candidate_snapshot=candidate_snapshot,
            previous_record_hash=inspection.record_hashes[-1],
            record_hash="0" * 64,
        )
        raw, record_hash = _record_json(record, "suzka.state-wal.transition")
        record = record.model_copy(update={"record_hash": record_hash})
        path = self._generation_path(inspection.active_manifest.active_generation_id)
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = self._open_private_dir(path.parent)
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "ab") as output:
                descriptor = -1
                self._validate_descriptor(output.fileno())
                output.write(_canonical(raw))
                output.flush()
                self._stage("transition_fsync")
                os.fsync(output.fileno())
            os.fsync(parent_descriptor)
        except Exception:
            raise StateWALError("WAL transition write failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
        return record

    def append_transition(self, **kwargs: Any) -> TransitionRecord:
        with self._lock:
            return self._append_transition(**kwargs)

    def publish_boot_anchor(
        self,
        *,
        snapshot_sequence: int,
        snapshot_hash: str,
        generation_id: UUID,
        anchored_record_id: UUID,
        anchored_record_hash: str,
        journal_processing_high_water: int,
        journal_tail_record_id: UUID | None = None,
        journal_tail_record_hash: str | None = None,
        journal_lineage_id: UUID,
    ) -> BootAnchor:
        # Under the WAL lock so that pruning's re-read of the boot anchor is
        # serialised against its publication (IP-3b review P2-5).
        with self._lock:
            return self._publish_boot_anchor_locked(
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
                generation_id=generation_id,
                anchored_record_id=anchored_record_id,
                anchored_record_hash=anchored_record_hash,
                journal_processing_high_water=journal_processing_high_water,
                journal_tail_record_id=journal_tail_record_id,
                journal_tail_record_hash=journal_tail_record_hash,
                journal_lineage_id=journal_lineage_id,
            )

    def _publish_boot_anchor_locked(
        self,
        *,
        snapshot_sequence: int,
        snapshot_hash: str,
        generation_id: UUID,
        anchored_record_id: UUID,
        anchored_record_hash: str,
        journal_processing_high_water: int,
        journal_tail_record_id: UUID | None,
        journal_tail_record_hash: str | None,
        journal_lineage_id: UUID,
    ) -> BootAnchor:
        _validate_sequence(snapshot_sequence, "snapshot sequence")
        _validate_sequence(
            journal_processing_high_water, "journal processing high-water"
        )
        inspection = self.inspect()
        if (
            inspection.active_manifest is None
            or generation_id != inspection.active_manifest.active_generation_id
        ):
            raise StateWALConflictError("boot anchor generation is not active")
        try:
            index = [record.record_id for record in inspection.records].index(
                anchored_record_id
            )
        except ValueError:
            raise StateWALConflictError("boot anchor record is not retained") from None
        record = inspection.records[index]
        snapshot = (
            record.baseline_snapshot
            if isinstance(record, BaselineRecord)
            else record.candidate_snapshot
        )
        if (
            snapshot.last_processed_event_sequence != snapshot_sequence
            or self._snapshot_hash_value(snapshot) != snapshot_hash
            or inspection.record_hashes[index] != anchored_record_hash
        ):
            raise StateWALIntegrityError("boot anchor does not match WAL")
        anchor_base = BootAnchor(
            schema_version=1,
            command_version=1,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            generation_id=generation_id,
            anchored_record_id=anchored_record_id,
            anchored_record_hash=anchored_record_hash,
            journal_processing_high_water=journal_processing_high_water,
            journal_tail_record_id=journal_tail_record_id,
            journal_tail_record_hash=journal_tail_record_hash,
            journal_lineage_id=journal_lineage_id,
            anchor_hash="0" * 64,
        )
        raw = anchor_base.model_dump(mode="json")
        raw["anchor_hash"] = _domain_hash(
            "suzka.state-wal.boot-anchor",
            {key: value for key, value in raw.items() if key != "anchor_hash"},
        )
        self._atomic_write(self.root / "boot_anchor.json", raw)
        return BootAnchor.model_validate_json(_canonical(raw))

    def inspect_boot_anchor(self) -> BootAnchor:
        try:
            raw = json.loads(self._read_regular(self.root / "boot_anchor.json"))
            anchor = BootAnchor.model_validate_json(_canonical(raw))
            expected = _domain_hash(
                "suzka.state-wal.boot-anchor",
                {key: value for key, value in raw.items() if key != "anchor_hash"},
            )
            if anchor.anchor_hash != expected:
                raise ValueError
            return anchor
        except Exception:
            raise StateWALIntegrityError("boot anchor is invalid") from None

    def inspect_boot_anchor_optional(self) -> BootAnchor | None:
        """Inspect an existing anchor without creating files."""

        if not self._prepare_for_read():
            return None
        try:
            (self.root / "boot_anchor.json").lstat()
        except FileNotFoundError:
            return None
        except OSError:
            raise StateWALIntegrityError("boot anchor cannot be inspected") from None
        return self.inspect_boot_anchor()

    def inspect_anchored_prefix(
        self,
    ) -> tuple[BootAnchor, CompatibleAgentStateSnapshot]:
        anchor = self.inspect_boot_anchor()
        snapshot = self.inspect_bound_prefix(
            generation_id=anchor.generation_id,
            record_id=anchor.anchored_record_id,
            record_hash=anchor.anchored_record_hash,
            snapshot_sequence=anchor.snapshot_sequence,
            snapshot_hash=anchor.snapshot_hash,
        )
        return anchor, snapshot

    def inspect_bound_prefix(
        self,
        *,
        generation_id: UUID,
        record_id: UUID,
        record_hash: str,
        snapshot_sequence: int,
        snapshot_hash: str,
    ) -> CompatibleAgentStateSnapshot:
        """Verify through an exact externally bound record, ignoring later bytes."""
        _validate_sequence(snapshot_sequence, "snapshot sequence")

        if not _HASH_RE.fullmatch(record_hash) or not _HASH_RE.fullmatch(snapshot_hash):
            raise StateWALFormatError("bound WAL identity is invalid")
        if not self._prepare_for_read():
            raise StateWALMissing("WAL is absent")
        path = self._generation_path(generation_id)
        lines = self._read_regular(path).splitlines(keepends=True)
        previous_hash: str | None = None
        previous_snapshot_sequence: int | None = None
        previous_state_hash: str | None = None
        previous_processing_sequence: int | None = None
        seen: set[UUID] = set()
        seen_events: set[UUID] = set()
        for line in lines:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError
                preflight_attention_json(line)
                raw = json.loads(line)
                kind = raw.get("record_type")
                model = (
                    BaselineRecord
                    if kind == "baseline"
                    else TransitionRecord
                    if kind == "transition"
                    else None
                )
                if model is None:
                    raise ValueError
                record = model.model_validate_json(_canonical(raw))
                computed_hash = _record_json(record, f"suzka.state-wal.{kind}")[1]
                if (
                    record.record_id in seen
                    or not _HASH_RE.fullmatch(record.record_hash)
                    or record.record_hash != computed_hash
                    or (
                        isinstance(record, TransitionRecord)
                        and record.event_id in seen_events
                    )
                ):
                    raise ValueError
                if record.generation_id != generation_id:
                    raise ValueError
                if not seen and not isinstance(record, BaselineRecord):
                    raise ValueError
                if isinstance(record, TransitionRecord):
                    if (
                        previous_snapshot_sequence is None
                        or previous_state_hash is None
                        or previous_processing_sequence is None
                        or record.previous_record_hash != previous_hash
                        or record.prior_snapshot_sequence != previous_snapshot_sequence
                        or record.prior_snapshot_hash != previous_state_hash
                        or record.processing_sequence <= previous_processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.processing_sequence
                        or record.candidate_snapshot_sequence
                        != record.candidate_snapshot.last_processed_event_sequence
                        or record.candidate_snapshot_hash
                        != self._snapshot_hash_value(record.candidate_snapshot)
                        or record.candidate_snapshot_sequence
                        <= previous_snapshot_sequence
                    ):
                        raise ValueError
                else:
                    if (
                        seen
                        or record.previous_record_hash is not None
                        or record.baseline_snapshot_sequence
                        != record.baseline_snapshot.last_processed_event_sequence
                        or record.baseline_snapshot_hash
                        != self._snapshot_hash_value(record.baseline_snapshot)
                        or record.journal_processing_high_water
                        < record.baseline_snapshot_sequence
                    ):
                        raise ValueError
                seen.add(record.record_id)
                if isinstance(record, TransitionRecord):
                    seen_events.add(record.event_id)
                snapshot = (
                    record.baseline_snapshot
                    if isinstance(record, BaselineRecord)
                    else record.candidate_snapshot
                )
                previous_snapshot_sequence = snapshot.last_processed_event_sequence
                previous_state_hash = (
                    record.candidate_snapshot_hash
                    if isinstance(record, TransitionRecord)
                    else record.baseline_snapshot_hash
                )
                previous_processing_sequence = (
                    record.processing_sequence
                    if isinstance(record, TransitionRecord)
                    else record.journal_processing_high_water
                )
                previous_hash = record.record_hash
                if record.record_id == record_id:
                    if (
                        computed_hash != record_hash
                        or snapshot.last_processed_event_sequence != snapshot_sequence
                        or self._snapshot_hash_value(snapshot) != snapshot_hash
                    ):
                        raise ValueError
                    return snapshot
            except RecursionError:
                raise StateWALIntegrityError("bound WAL prefix is invalid") from None
            except Exception:
                raise StateWALIntegrityError("bound WAL prefix is invalid") from None
        raise StateWALConflictError("bound WAL record is not retained")

    def reconstruct(
        self,
        *,
        sequence: int | None = None,
        snapshot_hash: str | None = None,
        record_id: UUID | None = None,
    ) -> CompatibleAgentStateSnapshot:
        if sequence is not None:
            _validate_sequence(sequence, "snapshot sequence")
        inspection = self._inspect_full()
        found: CompatibleAgentStateSnapshot | None = None
        for record in inspection.records:
            snapshot = (
                record.baseline_snapshot
                if isinstance(record, BaselineRecord)
                else record.candidate_snapshot
            )
            candidate_hash = (
                record.baseline_snapshot_hash
                if isinstance(record, BaselineRecord)
                else record.candidate_snapshot_hash
            )
            if (
                (sequence is None or snapshot.last_processed_event_sequence == sequence)
                and (snapshot_hash is None or candidate_hash == snapshot_hash)
                and (record_id is None or record.record_id == record_id)
            ):
                found = snapshot
        if found is None:
            raise StateWALConflictError("requested snapshot is not retained")
        return found

    def dry_run(
        self,
        *,
        sequence: int | None = None,
        snapshot_hash: str | None = None,
        record_id: UUID | None = None,
    ) -> DryRunDiff:
        inspection = self.inspect()
        current = self.reconstruct()
        target = self.reconstruct(
            sequence=sequence, snapshot_hash=snapshot_hash, record_id=record_id
        )
        before = current.model_dump(mode="json")
        after = target.model_dump(mode="json")
        changes = tuple(
            StateWALFieldChange(field=key, before=before.get(key), after=after.get(key))
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        )
        index = next(
            index
            for index, record in enumerate(inspection.records)
            if (
                record.baseline_snapshot
                if isinstance(record, BaselineRecord)
                else record.candidate_snapshot
            )
            == target
        )
        return DryRunDiff(
            target_snapshot_sequence=target.last_processed_event_sequence,
            target_snapshot_hash=self._snapshot_hash_value(target),
            current_snapshot_sequence=current.last_processed_event_sequence,
            current_snapshot_hash=self._snapshot_hash_value(current),
            changes=changes,
            records_replayed=index,
            external_side_effects_replayed=False,
        )
