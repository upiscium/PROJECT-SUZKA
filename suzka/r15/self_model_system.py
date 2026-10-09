"""U4 process-local SelfModel boundary with no trusted positive producer.

There is no joined finalized R07/R12 source plus Suzka-owned semantic/subject
admission, no typed task-attempt producer, and no R17/R18 verified outcome.
This owner can acknowledge a no-change request and read explicit UNKNOWN, but
cannot install caller-authored claims/capabilities or authenticate an EventRef.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Final, Iterator

from suzka.identifiers import validate_identifier
from suzka.limits import MAX_PERSISTED_REVISION
from suzka.r15.common import (
    Concept,
    EventRef,
    InterpretationStatus,
    SourceDisposition,
    SourceWitness,
    checksum,
    digest,
    encode,
    exact_enum,
    source_disposition,
)
from suzka.r15.contracts import (
    ClaimMeaning,
    SelfModelState,
    VerifiedCompetence,
    current_semantic_producer,
)


MAX_SELF_BOUNDARY_RECEIPTS: Final = 256
_REQUEST_DOMAIN: Final = b"PROJECT-SUZKA:R15:U4:BOUNDARY-REVIEW:V1\0"
_RECEIPT_DOMAIN: Final = b"PROJECT-SUZKA:R15:U4:BOUNDARY-RECEIPT:V1\0"


class SelfModelSystemError(ValueError):
    """Privacy-safe, bounded process-local error."""


class SelfModelEventConflict(SelfModelSystemError):
    """A stale, conflicting or reentrant EventRef cannot be acknowledged."""


class SelfModelCapacityExceeded(SelfModelSystemError):
    """No receipt eviction or silent shortening at process-local bounds."""


class SelfCapability(str, Enum):
    IDENTITY_HYPOTHESIS = "identity_hypothesis"
    ROLE_HYPOTHESIS = "role_hypothesis"
    POSSIBLE_TRAIT = "possible_trait"
    LIMITATION = "limitation"
    ATTEMPTED_TASK = "attempted_task"
    VERIFIED_COMPETENCE = "verified_competence"


@dataclass(frozen=True, slots=True)
class SelfAvailability:
    capability: SelfCapability
    source: SourceDisposition
    semantic: SourceDisposition
    admission: SourceDisposition
    verified: VerifiedCompetence

    def __post_init__(self) -> None:
        exact_enum(self.capability, SelfCapability, "capability")
        for name in ("source", "semantic", "admission"):
            exact_enum(getattr(self, name), SourceDisposition, name)
        if self.semantic is not SourceDisposition.UNAVAILABLE or self.admission is not SourceDisposition.UNAVAILABLE:
            raise ValueError("no current SelfModel semantic/subject admission producer exists")
        if self.capability is SelfCapability.VERIFIED_COMPETENCE and self.source is not SourceDisposition.UNAVAILABLE:
            raise ValueError("no R17/R18 verified-competence source exists")
        if type(self.verified) is not VerifiedCompetence or self.verified is not VerifiedCompetence.UNKNOWN:
            raise ValueError("R17/R18 verified competence is unavailable")


def self_availability(
    capability: SelfCapability,
    source: SourceWitness | None = None,
    meaning: ClaimMeaning | None = None,
) -> SelfAvailability:
    """Declared source requirement and negative semantic/outcome gate ONLY."""

    exact_enum(capability, SelfCapability, "capability")
    if source is not None and type(source) is not SourceWitness:
        raise TypeError("source must be an exact declared SourceWitness")
    if meaning is not None and type(meaning) is not ClaimMeaning:
        raise TypeError("meaning must be an exact closed ClaimMeaning")
    concept = (
        Concept.VERIFIED_COMPETENCE if capability is SelfCapability.VERIFIED_COMPETENCE
        else Concept.TASK_ATTEMPT if capability is SelfCapability.ATTEMPTED_TASK
        else Concept.SELF_HYPOTHESIS
    )
    source_status = (
        SourceDisposition.UNAVAILABLE if source is None
        else source_disposition(concept, source)
    )
    semantic_status = (
        SourceDisposition.UNAVAILABLE if meaning is None
        else current_semantic_producer(meaning)
    )
    return SelfAvailability(
        capability, source_status, semantic_status,
        SourceDisposition.UNAVAILABLE, VerifiedCompetence.UNKNOWN,
    )


@dataclass(frozen=True, slots=True)
class SelfModelReadView:
    revision: int
    root_digest: str
    claim_ids: tuple[str, ...]
    task_class_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelfTaskView:
    """UNKNOWN is neither attempted success nor a measured zero probability."""

    task_class_key: str
    tracked: bool
    attempted_status: InterpretationStatus
    attempted_confidence: int | None
    verified_competence: VerifiedCompetence

    def __post_init__(self) -> None:
        validate_identifier(self.task_class_key)
        if type(self.tracked) is not bool or self.tracked:
            raise ValueError("no current trusted task-class record may be tracked")
        if self.attempted_status is not InterpretationStatus.UNKNOWN or self.attempted_confidence is not None:
            raise ValueError("unverified task attempts remain UNKNOWN, not zero confidence")
        if self.verified_competence is not VerifiedCompetence.UNKNOWN:
            raise ValueError("verified competence is UNKNOWN until R17/R18")


@dataclass(frozen=True, slots=True)
class SelfBoundaryReceipt:
    """A request receipt, not attempt/outcome or personality evidence."""

    event: EventRef
    request_digest: str
    result_revision: int
    result_state_digest: str
    previous_receipt_digest: str | None

    def __post_init__(self) -> None:
        if type(self.event) is not EventRef:
            raise TypeError("receipt event must be exact")
        self.event.__post_init__()
        if type(self.result_revision) is not int or not 0 <= self.result_revision <= MAX_PERSISTED_REVISION:
            raise ValueError("receipt revision must be a bounded exact integer")
        digest(self.request_digest, "request_digest")
        digest(self.result_state_digest, "result_state_digest")
        if self.previous_receipt_digest is not None:
            digest(self.previous_receipt_digest, "previous_receipt_digest")

    @property
    def receipt_digest(self) -> str:
        self.__post_init__()
        return checksum(_RECEIPT_DOMAIN, encode(self))


@dataclass(frozen=True, slots=True)
class SelfBoundaryResult:
    snapshot: SelfModelState
    view: SelfModelReadView
    receipt: SelfBoundaryReceipt
    replayed: bool

    def __post_init__(self) -> None:
        if type(self.snapshot) is not SelfModelState or type(self.view) is not SelfModelReadView:
            raise TypeError("result root/view must be exact")
        if type(self.receipt) is not SelfBoundaryReceipt or type(self.replayed) is not bool:
            raise TypeError("result receipt/replay flag must be exact")
        self.receipt.__post_init__()
        root_digest = self.snapshot.canonical_value()["state_digest"]
        if (
            self.snapshot.revision != self.receipt.result_revision
            or root_digest != self.receipt.result_state_digest
            or self.view != _view(self.snapshot)
        ):
            raise ValueError("SelfModel receipt, root and view describe different results")


@dataclass(frozen=True, slots=True)
class SelfHistoricalAcknowledgement:
    """Original event receipt alone; never pair with a later current SelfModel."""

    receipt: SelfBoundaryReceipt
    replayed: bool = True

    def __post_init__(self) -> None:
        if type(self.receipt) is not SelfBoundaryReceipt or type(self.replayed) is not bool or not self.replayed:
            raise TypeError("historical acknowledgement needs exact replay receipt")
        self.receipt.__post_init__()

    @property
    def event_result_digest(self) -> str:
        return self.receipt.result_state_digest


def _copy_root(root: SelfModelState) -> SelfModelState:
    return SelfModelState.from_json(root.canonical_bytes())


def _copy_receipt(receipt: SelfBoundaryReceipt) -> SelfBoundaryReceipt:
    receipt.__post_init__()
    event = receipt.event
    return SelfBoundaryReceipt(
        EventRef(event.event_id, event.event_sequence, event.occurred_at),
        receipt.request_digest,
        receipt.result_revision,
        receipt.result_state_digest,
        receipt.previous_receipt_digest,
    )


def _view(root: SelfModelState) -> SelfModelReadView:
    return SelfModelReadView(
        root.revision, str(root.canonical_value()["state_digest"]),
        tuple(item.interpretation.statement_id for item in root.claims),
        tuple(item.task_class_key for item in root.capabilities),
    )


def _result_for(
    root: SelfModelState, receipt: SelfBoundaryReceipt, *, replayed: bool
) -> SelfBoundaryResult | SelfHistoricalAcknowledgement:
    """Pure result classification, NOT an install/admit-source method.

    A synthetic root used by tests cannot enter the production U4 owner.
    """

    if type(root) is not SelfModelState or type(receipt) is not SelfBoundaryReceipt:
        raise TypeError("replay classification needs exact root and receipt")
    receipt.__post_init__()
    if type(replayed) is not bool:
        raise TypeError("replayed must be exact")
    if root.revision != receipt.result_revision or root.canonical_value()["state_digest"] != receipt.result_state_digest:
        if not replayed:
            raise ValueError("fresh request cannot claim a different result root")
        return SelfHistoricalAcknowledgement(_copy_receipt(receipt))
    snapshot = _copy_root(root)
    return SelfBoundaryResult(snapshot, _view(snapshot), _copy_receipt(receipt), replayed)


class SelfModelSystem:
    """Process-local UNKNOWN-only root, no trait/task/capability writer."""

    def __init__(self) -> None:
        self._root = SelfModelState(1, 0, (), (), None, ())
        self._receipts: tuple[SelfBoundaryReceipt, ...] = ()
        self._lock = RLock()
        self._mutating = False
        self._reentrant_attempted = False

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        with self._lock:
            if self._mutating:
                self._reentrant_attempted = True
                raise SelfModelEventConflict("SelfModel boundary already processing")
            self._mutating = True
            self._reentrant_attempted = False
            try:
                yield
            finally:
                self._mutating = False
                self._reentrant_attempted = False

    def snapshot(self) -> SelfModelState:
        with self._lock:
            return _copy_root(self._root)

    def selected_view(self) -> SelfModelReadView:
        with self._lock:
            return _view(self._root)

    def task_view(self, task_class_key: str) -> SelfTaskView:
        """No task row is created merely by a read of an opaque class key."""

        key = validate_identifier(task_class_key)
        with self._lock:
            return SelfTaskView(key, False, InterpretationStatus.UNKNOWN, None, VerifiedCompetence.UNKNOWN)

    def review_boundary(self, event: EventRef) -> SelfBoundaryResult | SelfHistoricalAcknowledgement:
        """Acknowledge a no-change request, never an attempted task/outcome."""

        if type(event) is not EventRef:
            raise TypeError("event must be exact EventRef")
        event.__post_init__()
        event = EventRef(event.event_id, event.event_sequence, event.occurred_at)
        request_digest = checksum(_REQUEST_DOMAIN, {"event": encode(event), "operation": "availability_review"})
        with self._mutation():
            for receipt in self._receipts:
                if receipt.event.event_id == event.event_id or receipt.event.event_sequence == event.event_sequence:
                    if receipt.event != event or receipt.request_digest != request_digest:
                        raise SelfModelEventConflict("retained event identity or request conflicts")
                    result = _result_for(self._root, receipt, replayed=True)
                    if self._reentrant_attempted:
                        raise SelfModelEventConflict("reentrant replay invalidated the request")
                    return result
            if self._receipts:
                latest = self._receipts[-1].event
                if event.event_sequence <= latest.event_sequence or event.occurred_at < latest.occurred_at:
                    raise SelfModelEventConflict("unverifiable older or regressing event")
            if len(self._receipts) >= MAX_SELF_BOUNDARY_RECEIPTS:
                raise SelfModelCapacityExceeded("SelfModel request receipt window is full")
            root = self._root
            receipt = SelfBoundaryReceipt(
                event, request_digest, root.revision,
                str(root.canonical_value()["state_digest"]),
                self._receipts[-1].receipt_digest if self._receipts else None,
            )
            result = _result_for(root, receipt, replayed=False)
            assert isinstance(result, SelfBoundaryResult)
            if self._reentrant_attempted:
                raise SelfModelEventConflict("reentrant boundary request invalidated the event")
            self._receipts = self._receipts + (receipt,)
            return result
