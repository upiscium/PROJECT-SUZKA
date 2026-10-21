"""R15 U4: narrow source-qualified attempted-task SelfModel owner.

Only the internally bound serialized self-attempt gate can adopt the one
observed task occurrence. No verified competence, trait, identity, arbitrary
typed caller capability or model/operator truth setter exists.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Final, Iterator

from suzka.r15.attempt_producer import (
    TASK_CLASS,
    QualifiedSelfAttemptGate,
    QualifiedSelfAttemptUnavailable,
)
from suzka.identifiers import validate_identifier
from suzka.limits import MAX_PERSISTED_REVISION
from suzka.r15.common import (
    Concept,
    EventRef,
    Accessibility,
    Inertia,
    Interpretation,
    InterpretationStatus,
    SourceDisposition,
    SourceWitness,
    RevisionProof,
    RevisionReason,
    SCORE_SCALE,
    checksum,
    digest,
    encode,
    exact_enum,
    source_disposition,
)
from suzka.r15.contracts import (
    CapabilityHypothesis,
    ClaimMeaning,
    SelfModelState,
    VerifiedCompetence,
    current_semantic_producer,
)


MAX_SELF_BOUNDARY_RECEIPTS: Final = 256
_REQUEST_DOMAIN: Final = b"PROJECT-SUZKA:R15:U4:BOUNDARY-REVIEW:V1\0"
_RECEIPT_DOMAIN: Final = b"PROJECT-SUZKA:R15:U4:BOUNDARY-RECEIPT:V1\0"
_ATTEMPT_STATEMENT_DOMAIN: Final = b"PROJECT-SUZKA:R15:U4:OBSERVED-ATTEMPT:V1\0"
_ATTEMPT_CAPABILITY_DOMAIN: Final = b"PROJECT-SUZKA:R15:U4:ATTEMPT-CHANGE:V1\0"


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
        if (
            self.semantic is not SourceDisposition.UNAVAILABLE
            or self.admission is not SourceDisposition.UNAVAILABLE
        ):
            raise ValueError(
                "no current SelfModel semantic/subject admission producer exists"
            )
        if (
            self.capability is SelfCapability.VERIFIED_COMPETENCE
            and self.source is not SourceDisposition.UNAVAILABLE
        ):
            raise ValueError("no R17/R18 verified-competence source exists")
        if (
            type(self.verified) is not VerifiedCompetence
            or self.verified is not VerifiedCompetence.UNKNOWN
        ):
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
        Concept.VERIFIED_COMPETENCE
        if capability is SelfCapability.VERIFIED_COMPETENCE
        else Concept.TASK_ATTEMPT
        if capability is SelfCapability.ATTEMPTED_TASK
        else Concept.SELF_HYPOTHESIS
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
    return SelfAvailability(
        capability,
        source_status,
        semantic_status,
        SourceDisposition.UNAVAILABLE,
        VerifiedCompetence.UNKNOWN,
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
        if type(self.tracked) is not bool:
            raise TypeError("task tracking is a bounded exact bool")
        if self.tracked:
            if (
                self.attempted_status is not InterpretationStatus.PRESENT
                or type(self.attempted_confidence) is not int
                or not 0 <= self.attempted_confidence <= SCORE_SCALE
            ):
                raise ValueError("a tracked task requires an observed attempt")
        elif (
            self.attempted_status is not InterpretationStatus.UNKNOWN
            or self.attempted_confidence is not None
        ):
            raise ValueError("an untracked attempt remains UNKNOWN")
        if (
            type(self.verified_competence) is not VerifiedCompetence
            or self.verified_competence is not VerifiedCompetence.UNKNOWN
        ):
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
class SelfBoundaryResult:
    snapshot: SelfModelState
    view: SelfModelReadView
    receipt: SelfBoundaryReceipt
    replayed: bool

    def __post_init__(self) -> None:
        if (
            type(self.snapshot) is not SelfModelState
            or type(self.view) is not SelfModelReadView
        ):
            raise TypeError("result root/view must be exact")
        if (
            type(self.receipt) is not SelfBoundaryReceipt
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
            raise ValueError(
                "SelfModel receipt, root and view describe different results"
            )


@dataclass(frozen=True, slots=True)
class SelfHistoricalAcknowledgement:
    """Original event receipt alone; never pair with a later current SelfModel."""

    receipt: SelfBoundaryReceipt
    replayed: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not SelfBoundaryReceipt
            or type(self.replayed) is not bool
            or not self.replayed
        ):
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
        root.revision,
        str(root.canonical_value()["state_digest"]),
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
    if (
        root.revision != receipt.result_revision
        or root.canonical_value()["state_digest"] != receipt.result_state_digest
    ):
        if not replayed:
            raise ValueError("fresh request cannot claim a different result root")
        return SelfHistoricalAcknowledgement(_copy_receipt(receipt))
    snapshot = _copy_root(root)
    return SelfBoundaryResult(
        snapshot, _view(snapshot), _copy_receipt(receipt), replayed
    )


@dataclass(frozen=True, slots=True)
class SelfAttemptResult:
    """One actual internal Actor event; no task accuracy or competence proof."""

    snapshot: SelfModelState
    event: EventRef
    changed: bool
    task_class_key: str = TASK_CLASS
    verified_competence: VerifiedCompetence = VerifiedCompetence.UNKNOWN

    def __post_init__(self) -> None:
        if (
            type(self.snapshot) is not SelfModelState
            or type(self.event) is not EventRef
            or type(self.changed) is not bool
            or self.task_class_key != TASK_CLASS
            or self.verified_competence is not VerifiedCompetence.UNKNOWN
        ):
            raise ValueError("unqualified task-attempt result")


def _is_closed_admitted_attempt(
    root: SelfModelState, witness: SourceWitness | None = None
) -> bool:
    """Recognize only this internal producer's canonical task root.

    This structural check is NOT a current R07/R12 Source attestation. A
    restored legal U1 root with some other task cannot become a positive U4
    occurrence view merely by having the right field types.
    """
    if (
        type(root) is not SelfModelState
        or root.revision != 1
        or root.claims
        or len(root.capabilities) != 1
        or len(root.revision_history) != 1
    ):
        return False
    capability = root.capabilities[0]
    attempt = capability.attempted
    if len(attempt.support) != 1:
        return False
    evidence = attempt.support[0]
    if witness is not None and evidence != witness:
        return False
    expected_statement = checksum(
        _ATTEMPT_STATEMENT_DOMAIN,
        {"task_class": TASK_CLASS, "experience": evidence.reference},
    )
    proof = root.revision_history[0]
    return (
        capability.task_class_key == TASK_CLASS
        and capability.verified_competence is VerifiedCompetence.UNKNOWN
        and attempt.status is InterpretationStatus.PRESENT
        and attempt.statement_id == expected_statement
        and attempt.estimate == SCORE_SCALE
        and attempt.confidence == SCORE_SCALE
        and attempt.counter == ()
        and attempt.accessibility is Accessibility.CURRENT
        and attempt.inertia is Inertia.TENTATIVE
        and proof.target_id == expected_statement
        and proof.before_digest
        == str(SelfModelState(1, 0, (), (), None, ()).canonical_value()["state_digest"])
        and proof.after_digest
        == checksum(_ATTEMPT_CAPABILITY_DOMAIN, encode(capability))
        and proof.support_digests == (evidence.source_digest,)
        and proof.counter_digests == ()
        and proof.reason is RevisionReason.INITIAL_INTERPRETATION
        and proof.previous_digest is None
    )


class SelfModelSystem:
    """A single qualified task-attempt owner; never verified capability."""

    def __init__(self) -> None:
        self._root = SelfModelState(1, 0, (), (), None, ())
        self._attempt_gate: QualifiedSelfAttemptGate | None = None
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

    def _restore_technical_for_agent_state(self, state: SelfModelState) -> None:
        """U5-only exact root restoration; no traits or competence adopted."""
        if type(state) is not SelfModelState:
            raise TypeError("SelfModel technical state must be exact")
        detached = SelfModelState.from_json(state.canonical_bytes())
        with self._mutation():
            self._root = detached
            self._receipts = ()

    def selected_view(self) -> SelfModelReadView:
        with self._lock:
            return _view(self._root)

    def task_view(self, task_class_key: str) -> SelfTaskView:
        """Read the bounded producer-shaped task occurrence, not success."""

        key = validate_identifier(task_class_key)
        with self._lock:
            if key != TASK_CLASS or not _is_closed_admitted_attempt(self._root):
                return SelfTaskView(
                    key,
                    False,
                    InterpretationStatus.UNKNOWN,
                    None,
                    VerifiedCompetence.UNKNOWN,
                )
            attempted = self._root.capabilities[0]
            return SelfTaskView(
                key,
                True,
                attempted.attempted.status,
                attempted.attempted.confidence,
                attempted.verified_competence,
            )

    def _bind_source_qualified_attempt_gate(
        self, gate: QualifiedSelfAttemptGate
    ) -> None:
        """Internal one-time binding; never accepts a caller/model hypothesis."""

        if type(gate) is not QualifiedSelfAttemptGate:
            raise TypeError("attempt gate must be an exact bound Source owner")
        with self._mutation():
            if self._attempt_gate is not None:
                raise SelfModelEventConflict("SelfModel attempt gate already bound")
            self._attempt_gate = gate

    def _review_source_qualified_attempt(self) -> SelfAttemptResult:
        """Adopt only a real completed first-person E, in the Actor worker."""

        gate = self._attempt_gate
        if type(gate) is not QualifiedSelfAttemptGate:
            raise QualifiedSelfAttemptUnavailable("SelfModel source gate unavailable")
        with self._mutation():
            # No caller-supplied candidate, event, task class, source witness,
            # competence score, or operator/model JSON reaches this interface.
            observed = gate.qualify_current_attempt()
            current = self._root
            event = observed.review_event
            witness = observed.original_experience
            statement_id = checksum(
                _ATTEMPT_STATEMENT_DOMAIN,
                {"task_class": TASK_CLASS, "experience": witness.reference},
            )
            if current.revision_history:
                last = current.revision_history[-1].event
                if (
                    event.event_sequence <= last.event_sequence
                    or event.occurred_at < last.occurred_at
                ):
                    raise SelfModelEventConflict("SelfModel review predates revision")
            if current.revision != 0 or current.capabilities or current.claims:
                if not _is_closed_admitted_attempt(current, witness):
                    raise SelfModelEventConflict(
                        "different existing SelfModel cannot be overwritten"
                    )
                return SelfAttemptResult(_copy_root(current), event, False)

            attempt = Interpretation(
                statement_id,
                InterpretationStatus.PRESENT,
                SCORE_SCALE,
                SCORE_SCALE,
                (witness,),
                (),
                Accessibility.CURRENT,
                Inertia.TENTATIVE,
            )
            capability = CapabilityHypothesis(
                TASK_CLASS, attempt, VerifiedCompetence.UNKNOWN
            )
            proof = RevisionProof(
                1,
                event,
                attempt.statement_id,
                str(current.canonical_value()["state_digest"]),
                checksum(_ATTEMPT_CAPABILITY_DOMAIN, encode(capability)),
                (witness.source_digest,),
                (),
                RevisionReason.INITIAL_INTERPRETATION,
                None,
            )
            updated = SelfModelState(1, 1, (), (capability,), None, (proof,))
            detached = _copy_root(updated)
            if self._reentrant_attempted:
                raise SelfModelEventConflict("reentrant attempt invalidated review")
            # The enclosing Actor transaction's existing v10/WAL checkpoint
            # publishes this owner root; a failed handler restores it.
            self._root = detached
            return SelfAttemptResult(_copy_root(detached), event, True)

    def review_boundary(
        self, event: EventRef
    ) -> SelfBoundaryResult | SelfHistoricalAcknowledgement:
        """Acknowledge a no-change request, never an attempted task/outcome."""

        if type(event) is not EventRef:
            raise TypeError("event must be exact EventRef")
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
                        raise SelfModelEventConflict(
                            "retained event identity or request conflicts"
                        )
                    result = _result_for(self._root, receipt, replayed=True)
                    if self._reentrant_attempted:
                        raise SelfModelEventConflict(
                            "reentrant replay invalidated the request"
                        )
                    return result
            if self._receipts:
                latest = self._receipts[-1].event
                if (
                    event.event_sequence <= latest.event_sequence
                    or event.occurred_at < latest.occurred_at
                ):
                    raise SelfModelEventConflict(
                        "unverifiable older or regressing event"
                    )
            if len(self._receipts) >= MAX_SELF_BOUNDARY_RECEIPTS:
                raise SelfModelCapacityExceeded(
                    "SelfModel request receipt window is full"
                )
            root = self._root
            receipt = SelfBoundaryReceipt(
                event,
                request_digest,
                root.revision,
                str(root.canonical_value()["state_digest"]),
                self._receipts[-1].receipt_digest if self._receipts else None,
            )
            result = _result_for(root, receipt, replayed=False)
            assert isinstance(result, SelfBoundaryResult)
            if self._reentrant_attempted:
                raise SelfModelEventConflict(
                    "reentrant boundary request invalidated the event"
                )
            self._receipts = self._receipts + (receipt,)
            return result
