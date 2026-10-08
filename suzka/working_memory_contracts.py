"""Dependency-neutral Working Memory source contracts."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from suzka.context_contracts import ContextRelation
from suzka.identifiers import validate_identifier


MAX_ITEM_CAPACITY = 4_096
MAX_PROJECTION_BYTES = 16 * 1024 * 1024
MAX_SOURCE_ID_BYTES = 128
_ITEM_ID_DOMAIN = b"suzka-working-memory-item-v1\0"
_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class WorkingMemorySourceKind(str, Enum):
    """The only source authorities R08 U1 may reference."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class WorkingMemoryRetentionReason(str, Enum):
    """Small R08-only membership retention vocabulary."""

    RECENT = "recent"
    REACTIVATED = "reactivated"


class WorkingMemoryDecisionReason(str, Enum):
    """Bounded reasons emitted by pure projection selection."""

    SELECTED = "selected"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    RESOLVER_FAILURE = "resolver_failure"
    SOURCE_ARCHIVED = "source_archived"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_MALFORMED = "source_malformed"
    PROJECTION_BUDGET = "projection_budget"


class WorkingMemoryAdmissionReason(str, Enum):
    """Bounded result of one explicit admission mutation."""

    ADMITTED = "admitted"
    REACTIVATED = "reactivated"
    CAPACITY_EVICTED = "capacity_evicted"


class WorkingMemoryContextProjection(Protocol):
    """Opaque request-scoped Context evidence consumed only for ranking."""

    @property
    def aggregate_compatibility(self) -> float:
        """Return the already-computed aggregate Context compatibility."""
        ...


@dataclass(frozen=True, slots=True)
class WorkingMemoryItem:
    """Authoritative reference membership and R08-owned ranking metadata."""

    item_id: str
    source_kind: WorkingMemorySourceKind
    source_id: str
    activation: float
    salience: float
    retention_reason: WorkingMemoryRetentionReason
    created_revision: int
    last_activated_revision: int


@dataclass(frozen=True, slots=True)
class WorkingMemoryAdmission:
    """Explicitly distinguishes retained admission from immediate eviction."""

    item: WorkingMemoryItem
    retained: bool
    reason: WorkingMemoryAdmissionReason
    evicted_item_id: str | None


@dataclass(frozen=True, slots=True)
class WorkingMemorySelection:
    """One ephemeral resolved source selected for projection."""

    item_id: str
    source_kind: WorkingMemorySourceKind
    source_id: str
    rendered_content: str
    score: float
    reason: WorkingMemoryDecisionReason
    source_context_id: str | None = None
    context_relation: ContextRelation | None = None
    context_compatibility: float | None = None
    effective_score: float | None = None
    context_projection: WorkingMemoryContextProjection | None = None


@dataclass(frozen=True, slots=True)
class WorkingMemoryDecision:
    """One bounded selection decision, including unresolved/rejected items."""

    item_id: str
    source_kind: WorkingMemorySourceKind
    source_id: str
    selected: bool
    score: float
    reason: WorkingMemoryDecisionReason
    context_relation: ContextRelation | None = None
    context_compatibility: float | None = None
    effective_score: float | None = None
    context_projection: WorkingMemoryContextProjection | None = None


@dataclass(frozen=True, slots=True)
class WorkingMemoryView:
    """Immutable process-local projection; resolved content is not authority."""

    selected: tuple[WorkingMemorySelection, ...]
    decisions: tuple[WorkingMemoryDecision, ...]
    projected_bytes: int
    item_capacity: int
    projection_max_bytes: int
    revision: int


class WorkingMemoryResolutionStatus(str, Enum):
    """Bounded outcomes from resolving an authoritative source reference."""

    RESOLVED = "resolved"
    MISSING = "missing"
    ARCHIVED = "archived"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class WorkingMemoryResolution:
    """Immutable typed resolution; only resolved sources may carry content."""

    status: WorkingMemoryResolutionStatus
    rendered_content: str | None = None
    source_context_id: str | None = None
    context_projection: WorkingMemoryContextProjection | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not WorkingMemoryResolutionStatus:
            raise TypeError("status must be WorkingMemoryResolutionStatus")
        if self.status is WorkingMemoryResolutionStatus.RESOLVED:
            if type(self.rendered_content) is not str:
                raise ValueError("resolved Working Memory content must be a string")
            if self.source_context_id is not None:
                validate_identifier(self.source_context_id)
            if self.context_projection is not None:
                score = self.context_projection.aggregate_compatibility
                if (
                    type(score) is not float
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError("context projection compatibility is invalid")
        elif self.rendered_content is not None:
            raise ValueError("non-resolved Working Memory content must be None")
        elif self.source_context_id is not None or self.context_projection is not None:
            raise ValueError("non-resolved Working Memory provenance must be None")


WorkingMemoryResolverResult = str | WorkingMemoryResolution | None
WorkingMemoryResolver = Callable[[WorkingMemoryItem], WorkingMemoryResolverResult]


def _validate_source(
    source_kind: WorkingMemorySourceKind, source_id: str
) -> None:
    if type(source_kind) is not WorkingMemorySourceKind:
        raise TypeError("source_kind must be WorkingMemorySourceKind")
    if not isinstance(source_id, str):
        raise TypeError("source_id must be a string")
    if (
        not source_id
        or len(source_id.encode("utf-8")) > MAX_SOURCE_ID_BYTES
        or _SOURCE_ID.fullmatch(source_id) is None
        or source_id in {".", ".."}
    ):
        raise ValueError("source_id is not a valid bounded opaque identifier")


def working_memory_item_id(
    source_kind: WorkingMemorySourceKind, source_id: str
) -> str:
    """Derive stable identity as SHA-256(domain || kind || NUL || source ID)."""

    _validate_source(source_kind, source_id)
    payload = (
        _ITEM_ID_DOMAIN
        + source_kind.value.encode("ascii")
        + b"\0"
        + source_id.encode("ascii")
    )
    return "wm-" + hashlib.sha256(payload).hexdigest()
