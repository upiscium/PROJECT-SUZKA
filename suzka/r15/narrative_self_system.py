"""U3 process-local NarrativeSelf and narrow P1b-A source-qualified review.

Generic free-form biography, trust, traits and model/operator declarations
remain unavailable and cannot install any caller-made state. The optional
internal P1b-A source gate can qualify a *finite* self-referent interpretation
from real completed R07/R12/DB1 events in an anonymous R09 Context, but only
inside a distinct serialized Runtime event. No generic U3 semantic issuer,
AgentState v10/WAL persistence or production MainLoop consumer exists.
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
    Interpretation,
    InterpretationStatus,
    Accessibility,
    Inertia,
    RevisionProof,
    RevisionReason,
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
    NarrativeEpisode,
    NarrativeLink,
    NarrativeLinkKind,
    NarrativeSelfState,
    current_semantic_producer,
)
from suzka.r15.continuity_state import NarrativeContinuation, ContinuityInvalid
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
    scoped_option_key,
    restore_scoped_option,
)


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
        Concept.NARRATIVE_EPISODE
        if capability is NarrativeCapability.EPISODE_INDEX
        else Concept.NARRATIVE_CLAIM
    )
    source_status = (
        SourceDisposition.UNAVAILABLE
        if source is None
        else source_disposition(concept, source)
    )
    semantic_status = (
        SourceDisposition.UNAVAILABLE
        if meaning is None
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
        if (
            type(self.result_revision) is not int
            or not 0 <= self.result_revision <= MAX_PERSISTED_REVISION
        ):
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
        if (
            type(self.snapshot) is not NarrativeSelfState
            or type(self.view) is not NarrativeReadView
        ):
            raise TypeError("result root/view must be exact")
        if (
            type(self.receipt) is not NarrativeBoundaryReceipt
            or type(self.replayed) is not bool
        ):
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
        if (
            type(self.receipt) is not NarrativeBoundaryReceipt
            or type(self.replayed) is not bool
            or not self.replayed
        ):
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

    if (
        type(root) is not NarrativeSelfState
        or type(receipt) is not NarrativeBoundaryReceipt
    ):
        raise TypeError("replay classification needs exact root and receipt")
    receipt.__post_init__()
    if type(replayed) is not bool:
        raise TypeError("replayed must be an exact bool")
    if (
        root.revision != receipt.result_revision
        or root.canonical_value()["state_digest"] != receipt.result_state_digest
    ):
        if not replayed:
            raise ValueError("a fresh event cannot claim another result root")
        return NarrativeHistoricalAcknowledgement(_copy_receipt(receipt))
    snapshot = _copy_root(root)
    return NarrativeBoundaryResult(
        snapshot, _view(snapshot), _copy_receipt(receipt), replayed
    )


@dataclass(frozen=True, slots=True)
class NarrativeQualifiedResult:
    """Private process-local accepted *interpretation*, NOT verified intent."""

    snapshot: NarrativeSelfState
    event: EventRef
    stage: str
    selected_referent: str
    changed: bool
    source_scope: str


_NARRATIVE_EPISODE_DOMAIN: Final = b"PROJECT-SUZKA:R15:P1BA:EPISODE:V1\0"
_NARRATIVE_CHANGE_DOMAIN: Final = b"PROJECT-SUZKA:R15:P1BA:REVISION:V1\0"
_ZERO_DIGEST: Final = "0" * 64


def _source_episode_id(scope: str, stage: str) -> str:
    return checksum(_NARRATIVE_EPISODE_DOMAIN, {"scope": scope, "stage": stage})


def _source_episode_digest(episode: NarrativeEpisode) -> str:
    return checksum(_NARRATIVE_CHANGE_DOMAIN, encode(episode))


class NarrativeSelfSystem:
    """U3 owner: no general setter; narrowly-qualified internal P1b-A review."""

    def __init__(
        self, *, source_gate: QualifiedNarrativeSourceGate | None = None
    ) -> None:
        if (
            source_gate is not None
            and type(source_gate) is not QualifiedNarrativeSourceGate
        ):
            raise TypeError("invalid R15 source gate")
        self._source_gate = source_gate
        self._root = NarrativeSelfState(1, 0, (), (), None, ())
        self._receipts: tuple[NarrativeBoundaryReceipt, ...] = ()
        self._first_sources: tuple[QualifiedTurnRef, ...] | None = None
        self._revised_sources: tuple[QualifiedTurnRef, ...] | None = None
        self._source_context_id: str | None = None
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

    def _bind_source_gate_for_state(self, gate: QualifiedNarrativeSourceGate) -> None:
        """One-time internal startup binding, never a caller-supplied proof."""
        if type(gate) is not QualifiedNarrativeSourceGate:
            raise TypeError("source gate must be the owned runtime reader")
        with self._mutation():
            if self._source_gate is not None:
                raise NarrativeEventConflict("source gate already bound")
            self._source_gate = gate

    def _technical_state_for_agent_state(
        self,
    ) -> tuple[NarrativeSelfState, NarrativeContinuation]:
        """Export the U3 root and source replay scope under the SAME owner lock."""
        with self._lock:
            root = _copy_root(self._root)
            if root.revision == 0:
                return root, NarrativeContinuation.empty(root)
            if (
                self._first_sources is None
                or self._source_context_id is None
                or (root.revision == 2 and self._revised_sources is None)
            ):
                raise ContinuityInvalid("R15 owner lost qualified technical history")
            current = root.revision_history[-1].event
            state = NarrativeContinuation(
                str(root.canonical_value()["state_digest"]),
                root.revision,
                self._first_sources[0].episode_id,
                self._source_context_id,
                self._first_sources,
                self._revised_sources or (),
                current.event_id,
                current.event_sequence,
            )
            state.canonical_value(root)
            return root, state

    def _restore_technical_for_agent_state(
        self,
        root: NarrativeSelfState,
        continuation: NarrativeContinuation,
    ) -> None:
        """U5-only exact technical restore, not a general admission/setter."""
        if (
            type(root) is not NarrativeSelfState
            or type(continuation) is not NarrativeContinuation
        ):
            raise TypeError("technical root and continuation must be exact")
        detached = _copy_root(root)
        checked = NarrativeContinuation.from_value(
            continuation.canonical_value(detached), detached
        )
        with self._mutation():
            self._root = detached
            self._first_sources = checked.first_sources or None
            self._revised_sources = checked.revised_sources or None
            self._source_context_id = checked.context_id
            # No-change U3 acknowledgements were process-local and cannot
            # be reinterpreted as durable Experience/subject acceptance.
            self._receipts = ()

    def _review_committed_turns(
        self,
        sources: tuple[QualifiedTurnRef, ...],
    ) -> NarrativeQualifiedResult:
        """Internal-only MVI subject review, never a model/API source setter.

        The bound gate re-reads real terminal R07, R12, DB1 and anonymous R09
        inside an actual dedicated AgentRuntime worker event. An external
        model/user merely naming A/B or presenting an EventRef has no power to
        adopt any self/relationship/competence truth.
        """
        if type(sources) is not tuple or any(
            type(item) is not QualifiedTurnRef for item in sources
        ):
            raise QualifiedProposalUnavailable("unqualified source request")
        gate = self._source_gate
        if gate is None:
            raise QualifiedProposalUnavailable("no source owner bound")
        stage = "S1" if len(sources) == 3 else "S2" if len(sources) == 5 else ""
        if not stage:
            raise QualifiedProposalUnavailable("invalid narrative review stage")

        with self._mutation():
            review_event = gate.current_review_event()
            if self._receipts:
                last = self._receipts[-1].event
                if (
                    review_event.event_sequence <= last.event_sequence
                    or review_event.occurred_at < last.occurred_at
                ):
                    raise NarrativeEventConflict(
                        "narrative event predates boundary receipt"
                    )
            if self._root.revision_history:
                last = self._root.revision_history[-1].event
                if (
                    review_event.event_sequence <= last.event_sequence
                    or review_event.occurred_at < last.occurred_at
                ):
                    raise NarrativeEventConflict("narrative event did not advance")
            # This source read is NOT a source transfer of authority. The gate
            # independently confirms the ordinary Journal + Experience +
            # Episodic owners on each attempt, and returns only a finite option.
            proposal = gate.qualify(sources, stage=stage)
            if self._first_sources is not None and sources[:3] != self._first_sources:
                raise NarrativeEventConflict(
                    "different original first-person experience"
                )
            current = self._root
            if stage == "S1" and self._first_sources is not None:
                if current.revision != 1:
                    raise NarrativeEventConflict(
                        "old initial experience cannot replace revision"
                    )
                return NarrativeQualifiedResult(
                    _copy_root(current),
                    review_event,
                    stage,
                    restore_scoped_option(
                        proposal.source_scope,
                        current.episodes[0].interpretation.statement_id,
                    ),
                    False,
                    proposal.source_scope,
                )
            if stage == "S2":
                if self._first_sources is None or current.revision == 0:
                    raise NarrativeEventConflict("initial interpretation not admitted")
                if self._revised_sources is not None:
                    if sources != self._revised_sources:
                        raise NarrativeEventConflict(
                            "second reinterpretation needs a new stage"
                        )
                    return NarrativeQualifiedResult(
                        _copy_root(current),
                        review_event,
                        stage,
                        proposal.selected,
                        False,
                        proposal.source_scope,
                    )
            if stage == "S1" and current.revision != 0:
                raise NarrativeEventConflict(
                    "S1 admission requires initial empty state"
                )
            if stage == "S2" and current.revision != 1:
                raise NarrativeEventConflict(
                    "reinterpretation requires one prior revision"
                )

            episode_id = _source_episode_id(proposal.source_scope, stage)
            primary = proposal.source
            other = proposal.contrary
            episode = NarrativeEpisode(
                episode_id,
                primary,
                Interpretation(
                    scoped_option_key(proposal.source_scope, proposal.selected),
                    InterpretationStatus.PRESENT
                    if stage == "S1"
                    else InterpretationStatus.CONTESTED,
                    500_000,
                    250_000,
                    (primary,),
                    () if other is None else (other,),
                    Accessibility.CURRENT,
                    Inertia.TENTATIVE if stage == "S1" else Inertia.REVISABLE,
                ),
                ()
                if stage == "S1"
                else (
                    NarrativeLink(
                        NarrativeLinkKind.EPISODE,
                        _source_episode_id(proposal.source_scope, "S1"),
                    ),
                ),
            )
            old_proof = (
                current.revision_history[-1] if current.revision_history else None
            )
            previous_digest = (
                _source_episode_digest(current.episodes[0])
                if old_proof is not None
                else _ZERO_DIGEST
            )
            current_digest = _source_episode_digest(episode)
            revision = current.revision + 1
            proof = RevisionProof(
                revision,
                review_event,
                _source_episode_id(proposal.source_scope, "TARGET"),
                previous_digest,
                current_digest,
                (primary.source_digest,),
                () if other is None else (other.source_digest,),
                RevisionReason.INITIAL_INTERPRETATION
                if stage == "S1"
                else RevisionReason.QUALIFIED_REINTERPRETATION,
                None if old_proof is None else old_proof.record_digest,
            )
            candidate = NarrativeSelfState(
                1,
                revision,
                tuple(
                    sorted(
                        (*current.episodes, episode), key=lambda item: item.episode_id
                    )
                ),
                current.claims,
                current.history_anchor,
                (*current.revision_history, proof),
            )
            detached = _copy_root(candidate)
            if self._reentrant_attempted:
                raise NarrativeEventConflict(
                    "reentrant source review invalidates admission"
                )
            if stage == "S2" and self._source_context_id != proposal.context_id:
                raise NarrativeEventConflict("context changed across reinterpretation")
            # Publish the root only after all source/context checks complete.
            # The outer AgentState/WAL gate owns crash consistency.
            self._root = detached
            if stage == "S1":
                self._first_sources = sources
                self._source_context_id = proposal.context_id
            else:
                self._revised_sources = sources
            return NarrativeQualifiedResult(
                _copy_root(detached),
                review_event,
                stage,
                proposal.selected,
                True,
                proposal.source_scope,
            )

    def review_boundary(
        self, event: EventRef
    ) -> NarrativeBoundaryResult | NarrativeHistoricalAcknowledgement:
        """Acknowledge no-change ordered *request*, never an Experience event."""

        if type(event) is not EventRef:
            raise TypeError("event must be an exact EventRef")
        event.__post_init__()
        event = EventRef(event.event_id, event.event_sequence, event.occurred_at)
        request_digest = checksum(
            _REQUEST_DOMAIN,
            {"event": encode(event), "operation": "availability_review"},
        )
        with self._mutation():
            for receipt in self._receipts:
                if (
                    receipt.event.event_id == event.event_id
                    or receipt.event.event_sequence == event.event_sequence
                ):
                    if (
                        receipt.event != event
                        or receipt.request_digest != request_digest
                    ):
                        raise NarrativeEventConflict(
                            "retained event identity or operation conflicts"
                        )
                    result = _result_for(self._root, receipt, replayed=True)
                    if self._reentrant_attempted:
                        raise NarrativeEventConflict(
                            "reentrant replay invalidated the request"
                        )
                    return result
            if self._receipts:
                latest = self._receipts[-1].event
                if (
                    event.event_sequence <= latest.event_sequence
                    or event.occurred_at < latest.occurred_at
                ):
                    raise NarrativeEventConflict(
                        "unverifiable older or regressing event"
                    )
            if len(self._receipts) >= MAX_NARRATIVE_BOUNDARY_RECEIPTS:
                raise NarrativeCapacityExceeded(
                    "Narrative request receipt window is full"
                )
            root = self._root
            receipt = NarrativeBoundaryReceipt(
                event,
                request_digest,
                root.revision,
                str(root.canonical_value()["state_digest"]),
                self._receipts[-1].receipt_digest if self._receipts else None,
            )
            result = _result_for(root, receipt, replayed=False)
            assert isinstance(result, NarrativeBoundaryResult)
            if self._reentrant_attempted:
                raise NarrativeEventConflict(
                    "reentrant boundary request invalidated the event"
                )
            self._receipts = self._receipts + (receipt,)
            return result
