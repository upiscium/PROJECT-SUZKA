"""U3 process-local NarrativeSelf boundary with NO positive source producer.

Neither R07/R12's joined finalized Experience nor an R15 subject-owned
semantic/admission producer is available as an independently side-effect-free
read. This owner can review an event boundary without changing its empty U1
root and can report explicit unavailability; it cannot install a caller-made
episode/claim, consume a SourceWitness as proof or authenticate EventRef.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Final, Iterator

from suzka.limits import MAX_PERSISTED_REVISION
from suzka.r15.common import (
    Concept,
    EventRef,
    SourceDisposition,
    SourceWitness,
    checksum,
    digest,
    encode,
    exact_enum,
    source_disposition,
)
from suzka.r15.contracts import ClaimMeaning, NarrativeSelfState, current_semantic_producer


MAX_NARRATIVE_BOUNDARY_RECEIPTS: Final = 256
_REQUEST_DOMAIN: Final = b"PROJECT-SUZKA:R15:U3:BOUNDARY-REVIEW:V1\0"
_RECEIPT_DOMAIN: Final = b"PROJECT-SUZKA:R15:U3:BOUNDARY-RECEIPT:V1\0"


class NarrativeSystemError(ValueError):
    """Fail-closed private-free U3 boundary failure."""


class NarrativeEventConflict(NarrativeSystemError):
    """Event identity, chronology or reentrant review conflicts."""


class NarrativeCapacityExceeded(NarrativeSystemError):
    """No silent eviction of retained process-local acknowledgements."""


class NarrativeCapability(str, Enum):
    EPISODE_INDEX = "episode_index"
    CLAIM_ADMISSION = "claim_admission"
    INTERPRETATION_REVISION = "interpretation_revision"
    ACCESSIBILITY_REACTIVATION = "accessibility_reactivation"


@dataclass(frozen=True, slots=True)
class NarrativeAvailability:
    """A source-shaped tuple can require proof; that is not an admission."""

    capability: NarrativeCapability
    source: SourceDisposition
    semantic: SourceDisposition
    admission: SourceDisposition

    def __post_init__(self) -> None:
        exact_enum(self.capability, NarrativeCapability, "capability")
        for name in ("source", "semantic", "admission"):
            exact_enum(getattr(self, name), SourceDisposition, name)
        if self.admission is not SourceDisposition.UNAVAILABLE:
            raise ValueError("no U3 production subject admission exists")
        if self.semantic is not SourceDisposition.UNAVAILABLE:
            raise ValueError("no R14 semantic narrative producer exists")


def narrative_availability(
    capability: NarrativeCapability,
    source: SourceWitness | None = None,
    meaning: ClaimMeaning | None = None,
) -> NarrativeAvailability:
    """Read-only typed refusal, never an Experience read or transition."""

    exact_enum(capability, NarrativeCapability, "capability")
    if source is not None and type(source) is not SourceWitness:
        raise TypeError("source must be an exact declared SourceWitness")
    if meaning is not None and type(meaning) is not ClaimMeaning:
        raise TypeError("meaning must be an exact closed ClaimMeaning")
    concept = (
        Concept.NARRATIVE_EPISODE if capability is NarrativeCapability.EPISODE_INDEX
        else Concept.NARRATIVE_CLAIM
    )
    source_status = (
        SourceDisposition.UNAVAILABLE if source is None
        else source_disposition(concept, source)
    )
    semantic_status = (
        SourceDisposition.UNAVAILABLE if meaning is None
        else current_semantic_producer(meaning)
    )
    return NarrativeAvailability(
        capability, source_status, semantic_status, SourceDisposition.UNAVAILABLE
    )


@dataclass(frozen=True, slots=True)
class NarrativeReadView:
    revision: int
    root_digest: str
    episode_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NarrativeBoundaryReceipt:
    """No-change event REQUEST receipt, not R12 finality or autobiography."""

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
class NarrativeBoundaryResult:
    """A receipt paired only with the very root and view it describes."""

    snapshot: NarrativeSelfState
    view: NarrativeReadView
    receipt: NarrativeBoundaryReceipt
    replayed: bool

    def __post_init__(self) -> None:
        if type(self.snapshot) is not NarrativeSelfState or type(self.view) is not NarrativeReadView:
            raise TypeError("result root/view must be exact")
        if type(self.receipt) is not NarrativeBoundaryReceipt or type(self.replayed) is not bool:
            raise TypeError("result receipt/replay flag must be exact")
        self.receipt.__post_init__()
        root_digest = self.snapshot.canonical_value()["state_digest"]
        if (
            self.snapshot.revision != self.receipt.result_revision
            or root_digest != self.receipt.result_state_digest
            or self.view != _view(self.snapshot)
        ):
            raise ValueError("Narrative receipt, root and view are different results")


@dataclass(frozen=True, slots=True)
class NarrativeHistoricalAcknowledgement:
    """Original event receipt only; never attaches a later current root/view."""

    receipt: NarrativeBoundaryReceipt
    replayed: bool = True

    def __post_init__(self) -> None:
        if type(self.receipt) is not NarrativeBoundaryReceipt or type(self.replayed) is not bool or not self.replayed:
            raise TypeError("historical acknowledgement needs an exact replay receipt")
        self.receipt.__post_init__()

    @property
    def event_result_digest(self) -> str:
        return self.receipt.result_state_digest


def _copy_root(root: NarrativeSelfState) -> NarrativeSelfState:
    return NarrativeSelfState.from_json(root.canonical_bytes())


def _copy_receipt(receipt: NarrativeBoundaryReceipt) -> NarrativeBoundaryReceipt:
    receipt.__post_init__()
    event = receipt.event
    return NarrativeBoundaryReceipt(
        EventRef(event.event_id, event.event_sequence, event.occurred_at),
        receipt.request_digest,
        receipt.result_revision,
        receipt.result_state_digest,
        receipt.previous_receipt_digest,
    )


def _view(root: NarrativeSelfState) -> NarrativeReadView:
    return NarrativeReadView(
        root.revision,
        str(root.canonical_value()["state_digest"]),
        tuple(item.episode_id for item in root.episodes),
        tuple(item.interpretation.statement_id for item in root.claims),
    )


def _result_for(
    root: NarrativeSelfState, receipt: NarrativeBoundaryReceipt, *, replayed: bool
) -> NarrativeBoundaryResult | NarrativeHistoricalAcknowledgement:
    """Pure classification, not an admission or a root-installation API.

    Test-only synthetic U1 roots may exercise historic mismatch classification;
    no production method can install such a root until separately reviewed.
    """

    if type(root) is not NarrativeSelfState or type(receipt) is not NarrativeBoundaryReceipt:
        raise TypeError("replay classification needs exact root and receipt")
    receipt.__post_init__()
    if type(replayed) is not bool:
        raise TypeError("replayed must be an exact bool")
    if root.revision != receipt.result_revision or root.canonical_value()["state_digest"] != receipt.result_state_digest:
        if not replayed:
            raise ValueError("a fresh event cannot claim another result root")
        return NarrativeHistoricalAcknowledgement(_copy_receipt(receipt))
    snapshot = _copy_root(root)
    return NarrativeBoundaryResult(snapshot, _view(snapshot), _copy_receipt(receipt), replayed)


class NarrativeSelfSystem:
    """No source/claim setter: U3 only owns process-local UNKNOWN absence."""

    def __init__(self) -> None:
        self._root = NarrativeSelfState(1, 0, (), (), None, ())
        self._receipts: tuple[NarrativeBoundaryReceipt, ...] = ()
        self._lock = RLock()
        self._mutating = False
        self._reentrant_attempted = False

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        with self._lock:
            if self._mutating:
                self._reentrant_attempted = True
                raise NarrativeEventConflict("Narrative boundary already processing")
            self._mutating = True
            self._reentrant_attempted = False
            try:
                yield
            finally:
                self._mutating = False
                self._reentrant_attempted = False

    def snapshot(self) -> NarrativeSelfState:
        with self._lock:
            return _copy_root(self._root)

    def selected_view(self) -> NarrativeReadView:
        with self._lock:
            return _view(self._root)

    def review_boundary(
        self, event: EventRef
    ) -> NarrativeBoundaryResult | NarrativeHistoricalAcknowledgement:
        """Acknowledge no-change ordered *request*, never an Experience event."""

        if type(event) is not EventRef:
            raise TypeError("event must be an exact EventRef")
        event.__post_init__()
        event = EventRef(event.event_id, event.event_sequence, event.occurred_at)
        request_digest = checksum(_REQUEST_DOMAIN, {"event": encode(event), "operation": "availability_review"})
        with self._mutation():
            for receipt in self._receipts:
                if receipt.event.event_id == event.event_id or receipt.event.event_sequence == event.event_sequence:
                    if receipt.event != event or receipt.request_digest != request_digest:
                        raise NarrativeEventConflict("retained event identity or operation conflicts")
                    result = _result_for(self._root, receipt, replayed=True)
                    if self._reentrant_attempted:
                        raise NarrativeEventConflict("reentrant replay invalidated the request")
                    return result
            if self._receipts:
                latest = self._receipts[-1].event
                if event.event_sequence <= latest.event_sequence or event.occurred_at < latest.occurred_at:
                    raise NarrativeEventConflict("unverifiable older or regressing event")
            if len(self._receipts) >= MAX_NARRATIVE_BOUNDARY_RECEIPTS:
                raise NarrativeCapacityExceeded("Narrative request receipt window is full")
            root = self._root
            receipt = NarrativeBoundaryReceipt(
                event, request_digest, root.revision,
                str(root.canonical_value()["state_digest"]),
                self._receipts[-1].receipt_digest if self._receipts else None,
            )
            result = _result_for(root, receipt, replayed=False)
            assert isinstance(result, NarrativeBoundaryResult)
            if self._reentrant_attempted:
                raise NarrativeEventConflict("reentrant boundary request invalidated the event")
            self._receipts = self._receipts + (receipt,)
            return result
