"""R15 U2 process-local authority for *unknown* R09 interlocutor continuity.

This owner does not verify persons, Experience, emotion causation or Suzka's
subjective relationship interpretations. No production caller is wired yet:
the future U6 ordered runtime must authenticate the supplied EventRef and hold
the R09 read stable across its transaction. All positive axis admissions are
unavailable until a separately reviewed producer exists. Receipts are local
to this instance, never durable AgentState/WAL or an admission credential.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Final, Iterator

from suzka.r15.interaction_producer import (
    QualifiedInteractionSourceGate,
    QualifiedInteractionUnavailable,
)

from suzka.limits import MAX_PERSISTED_REVISION
from suzka.r15.common import (
    MAX_RELATIONSHIPS,
    MAX_REVISIONS,
    Accessibility,
    EventRef,
    Inertia,
    SCORE_SCALE,
    Interpretation,
    InterpretationStatus,
    RevisionProof,
    RevisionReason,
    SourceDisposition,
    SourceWitness,
    checksum,
    digest,
    encode,
    exact_enum,
    identifier,
)
from suzka.r15.contracts import (
    RelationshipAxis,
    RelationshipAxisKind,
    RelationshipRecord,
    RelationshipState,
)
from suzka.runtime.context import (
    ContextFrame,
    ContextRegistry,
    ContextRegistryState,
    ContextStatus,
    validate_context_registry_state,
)


MAX_OBSERVATION_RECEIPTS: Final = 256
_IDENTITY_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2:OPAQUE-RELATIONSHIP:V1\0"
_AXIS_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2:RELATIONSHIP-AXIS:V1\0"
_PROJECTION_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2:RELATIONSHIP-PROJECTION:V1\0"
_R09_PROJECTION_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2:SELECTED-R09:V1\0"
_R09_AUTHORITY_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2:R09-AUTHORITY:V1\0"
_REQUEST_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2:OBSERVE-REQUEST:V1\0"
_RECEIPT_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2:OBSERVATION-RECEIPT:V1\0"
_COORDINATION_Q1_DOMAIN: Final = b"PROJECT-SUZKA:R15:U2B:JOINT-Q1-PENDING:V1\0"


class RelationshipSystemError(ValueError):
    """Bounded, privacy-safe failure before a process-local publication."""


class RelationshipSourceUnavailable(RelationshipSystemError):
    """Current R09 participant cannot be independently located."""


class RelationshipEventConflict(RelationshipSystemError):
    """An event/retry disagrees with retained identity or ordering."""


class RelationshipCapacityExceeded(RelationshipSystemError):
    """The record, proof or process-local receipt window is full."""


class RelationshipAuthority(str, Enum):
    POSITIVE_AXIS_INTERPRETATION = "positive_axis_interpretation"
    VERIFIED_PERSON_MERGE = "verified_person_merge"
    ACTOR_CAUSATION = "actor_causation"


def relationship_authority_status(
    authority: RelationshipAuthority,
) -> SourceDisposition:
    """Generic caller/identity/trait authority is unavailable.

    U2b has only a closed, instance-bound, internal Actor Q1 coordination
    review. This global static query does not authenticate a source, grant
    arbitrary axis edits or claim verified person/causation authority.
    """

    exact_enum(authority, RelationshipAuthority, "relationship authority")
    return SourceDisposition.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class RelationshipAxisView:
    kind: RelationshipAxisKind
    status: InterpretationStatus
    estimate: int | None
    confidence: int | None
    accessibility: Accessibility
    inertia: Inertia


@dataclass(frozen=True, slots=True)
class RelationshipSelectedView:
    """Detached process-local read; no source payload or speculative result."""

    root_revision: int
    root_digest: str
    relationship_id: str
    interlocutor_key: str
    axes: tuple[RelationshipAxisView, ...]


@dataclass(frozen=True, slots=True)
class RelationshipObservationReceipt:
    """Local replay identity, NOT evidence of subject adoption or source finality."""

    event: EventRef
    interlocutor_key: str
    context_id: str
    context_revision: int
    context_digest: str
    request_digest: str
    before_state_digest: str
    result_state_digest: str
    result_revision: int
    created: bool
    previous_receipt_digest: str | None

    def __post_init__(self) -> None:
        if type(self.event) is not EventRef:
            raise TypeError("receipt event must be exact")
        self.event.__post_init__()
        identifier(self.interlocutor_key, "interlocutor_key")
        identifier(self.context_id, "context_id")
        if (
            type(self.context_revision) is not int
            or not 0 <= self.context_revision <= MAX_PERSISTED_REVISION
        ):
            raise ValueError("receipt Context revision is invalid")
        if (
            type(self.result_revision) is not int
            or not 0 <= self.result_revision <= MAX_PERSISTED_REVISION
        ):
            raise ValueError("receipt Relationship revision is invalid")
        for name in (
            "context_digest",
            "request_digest",
            "before_state_digest",
            "result_state_digest",
        ):
            digest(getattr(self, name), name)
        if self.previous_receipt_digest is not None:
            digest(self.previous_receipt_digest, "previous receipt digest")
        if type(self.created) is not bool:
            raise TypeError("created must be an exact bool")

    @property
    def receipt_digest(self) -> str:
        self.__post_init__()
        return checksum(_RECEIPT_DOMAIN, encode(self))


@dataclass(frozen=True, slots=True)
class RelationshipObservationResult:
    """One event receipt paired ONLY with its exact resulting root/view."""

    snapshot: RelationshipState
    receipt: RelationshipObservationReceipt
    replayed: bool
    view: RelationshipSelectedView

    def __post_init__(self) -> None:
        if (
            type(self.snapshot) is not RelationshipState
            or type(self.receipt) is not RelationshipObservationReceipt
        ):
            raise TypeError("event result requires exact root and receipt")
        if (
            type(self.view) is not RelationshipSelectedView
            or type(self.replayed) is not bool
        ):
            raise TypeError("event result requires exact view and replay flag")
        self.receipt.__post_init__()
        root_digest = self.snapshot.canonical_value()["state_digest"]
        if (
            self.snapshot.revision != self.receipt.result_revision
            or root_digest != self.receipt.result_state_digest
            or self.view.root_revision != self.snapshot.revision
            or self.view.root_digest != root_digest
            or self.view.interlocutor_key != self.receipt.interlocutor_key
        ):
            raise ValueError(
                "event receipt, resulting root and view do not bind the same revision"
            )
        record = next(
            (
                item
                for item in self.snapshot.records
                if item.interlocutor_key == self.receipt.interlocutor_key
            ),
            None,
        )
        if (
            record is None
            or self.view.relationship_id != record.relationship_id
            or self.view.axes
            != tuple(
                RelationshipAxisView(
                    axis.kind,
                    axis.interpretation.status,
                    axis.interpretation.estimate,
                    axis.interpretation.confidence,
                    axis.interpretation.accessibility,
                    axis.interpretation.inertia,
                )
                for axis in record.axes
            )
        ):
            raise ValueError("event view does not describe the receipted relationship")


@dataclass(frozen=True, slots=True)
class RelationshipHistoricalAcknowledgement:
    """Past event proof ONLY: no historical root retained, no current view implied.

    Call snapshot()/selected_view() separately for the latest state. Neither
    is the historical event result and must not be paired with this receipt.
    """

    receipt: RelationshipObservationReceipt
    replayed: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not RelationshipObservationReceipt
            or type(self.replayed) is not bool
            or not self.replayed
        ):
            raise TypeError(
                "historical acknowledgement requires an exact replay receipt"
            )
        self.receipt.__post_init__()

    @property
    def event_result_digest(self) -> str:
        """Original event's root checksum, not a current-state checksum."""

        return self.receipt.result_state_digest


def _projection_digest(revision: int, records: tuple[RelationshipRecord, ...]) -> str:
    """A root's current data projection, excluding proof history/root checksum.

    RevisionProof.after_digest cannot refer to RelationshipState.state_digest:
    the latter includes that very proof. U5 must preserve this declared scope.
    """

    return checksum(
        _PROJECTION_DOMAIN, {"revision": revision, "records": encode(records)}
    )


def _unknown_record(interlocutor_key: str, revision: int) -> RelationshipRecord:
    relationship_id = checksum(_IDENTITY_DOMAIN, {"interlocutor_key": interlocutor_key})
    axes = tuple(
        RelationshipAxis(
            kind,
            Interpretation(
                checksum(
                    _AXIS_DOMAIN,
                    {"relationship_id": relationship_id, "axis": kind.value},
                ),
                InterpretationStatus.UNKNOWN,
                None,
                None,
                (),
                (),
                Accessibility.CURRENT,
                Inertia.TENTATIVE,
            ),
        )
        for kind in sorted(RelationshipAxisKind, key=lambda item: item.value)
    )
    return RelationshipRecord(relationship_id, interlocutor_key, None, revision, axes)


def _current_frame(
    state: ContextRegistryState, event: EventRef, key: str
) -> ContextFrame:
    if type(state) is not ContextRegistryState:
        raise RelationshipSourceUnavailable("R09 current projection is unavailable")
    try:
        validate_context_registry_state(state)
    except ValueError:
        raise RelationshipSourceUnavailable(
            "R09 current projection is invalid"
        ) from None
    frame = next(
        (item for item in state.frames if item.context_id == state.current_context_id),
        None,
    )
    if (
        frame is None
        or frame.status is not ContextStatus.ACTIVE
        or key not in frame.participant_refs
    ):
        raise RelationshipSourceUnavailable("R09 current participant is unavailable")
    if frame.started_at > event.occurred_at or frame.last_active_at > event.occurred_at:
        raise RelationshipSourceUnavailable(
            "R09 participant is not current at this event time"
        )
    return frame


def _r09_digest(
    state: ContextRegistryState, frame: ContextFrame, event: EventRef
) -> str:
    # No R09 identity_key/confidence; this is an opaque participant projection.
    return checksum(
        _R09_PROJECTION_DOMAIN,
        {
            "event": encode(event),
            "registry_revision": state.revision,
            "context_id": frame.context_id,
            "frame_revision": frame.last_modified_revision,
            "started_at": encode(frame.started_at),
            "last_active_at": encode(frame.last_active_at),
            "participants": list(frame.participant_refs),
            "status": frame.status.value,
        },
    )


def _r09_authority_digest(state: ContextRegistryState) -> str:
    """Checksum one complete source generation, not an authenticated signature."""

    return checksum(_R09_AUTHORITY_DOMAIN, encode(state))


@dataclass(frozen=True, slots=True)
class RelationshipIssueReviewResult:
    """Actual internal Actor result, not verified person or trust."""

    snapshot: RelationshipState
    event: EventRef
    changed: bool
    interlocutor_key: str

    def __post_init__(self) -> None:
        if (
            type(self.snapshot) is not RelationshipState
            or type(self.event) is not EventRef
            or type(self.changed) is not bool
        ):
            raise ValueError("invalid bounded Relationship review")


def _source_qualified_q1_root(
    root: RelationshipState, key: str, witness: SourceWitness
) -> bool:
    """Producer-shaped historical proof, not a CURRENT Source re-read."""

    if (
        type(root) is not RelationshipState
        or root.revision != 1
        or root.history_anchor is not None
        or len(root.records) != 1
        or len(root.revision_history) != 1
    ):
        return False
    record = root.records[0]
    if (
        record.interlocutor_key != key
        or record.claimed_identity_key is not None
        or record.revision != 1
    ):
        return False
    original = _unknown_record(key, 1)
    q1_axis = next(
        (
            axis
            for axis in record.axes
            if axis.kind is RelationshipAxisKind.UNRESOLVED_ISSUE
        ),
        None,
    )
    if q1_axis is None:
        return False
    meaning = q1_axis.interpretation
    target_id = checksum(
        _COORDINATION_Q1_DOMAIN,
        {"interlocutor_key": key, "experience": witness.reference},
    )
    if (
        record.relationship_id != original.relationship_id
        or meaning.statement_id != target_id
        or meaning.status is not InterpretationStatus.PRESENT
        or meaning.estimate != SCORE_SCALE
        or meaning.confidence != SCORE_SCALE
        or meaning.support != (witness,)
        or meaning.counter
        or meaning.accessibility is not Accessibility.CURRENT
        or meaning.inertia is not Inertia.TENTATIVE
        or record.axes
        != tuple(
            RelationshipAxis(axis.kind, meaning)
            if axis.kind is RelationshipAxisKind.UNRESOLVED_ISSUE
            else axis
            for axis in original.axes
        )
    ):
        return False
    proof = root.revision_history[0]
    if witness.event is None:
        return False
    return (
        proof.revision == 1
        and proof.target_id == target_id
        and proof.event.event_sequence > witness.event.event_sequence
        and proof.event.occurred_at >= witness.event.occurred_at
        and proof.before_digest == _projection_digest(0, ())
        and proof.after_digest == _projection_digest(1, root.records)
        and proof.support_digests == (witness.source_digest,)
        and proof.counter_digests == ()
        and proof.reason is RevisionReason.INITIAL_INTERPRETATION
        and proof.previous_digest is None
    )


class RelationshipSystem:
    """One process-local R15 owner; no direct intrinsic adjustment method."""

    def __init__(self, context_registry: ContextRegistry) -> None:
        if type(context_registry) is not ContextRegistry:
            raise TypeError("Relationship owner requires the exact R09 ContextRegistry")
        self._registry = context_registry
        self._root = RelationshipState(1, 0, (), None, ())
        self._q1_gate: QualifiedInteractionSourceGate | None = None
        self._receipts: tuple[RelationshipObservationReceipt, ...] = ()
        # The full R09 checksum can correlate private declarative bindings.
        # Retain it ONLY internally for same-revision fork detection; neither
        # receipts nor selected views expose this whole-registry fingerprint.
        self._last_registry_digest: str | None = None
        self._lock = RLock()
        self._mutating = False
        self._reentrant_attempted = False

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        with self._lock:
            if self._mutating:
                self._reentrant_attempted = True
                raise RelationshipEventConflict(
                    "relationship mutation is already in progress"
                )
            self._mutating = True
            self._reentrant_attempted = False
            try:
                yield
            finally:
                self._mutating = False
                self._reentrant_attempted = False

    def _copy_root(self, root: RelationshipState) -> RelationshipState:
        # Validate and detach every nested immutable field; never hand out the
        # owner instance as a speculative or externally editable root.
        return RelationshipState.from_json(root.canonical_bytes())

    @staticmethod
    def _copy_receipt(
        receipt: RelationshipObservationReceipt,
    ) -> RelationshipObservationReceipt:
        receipt.__post_init__()
        source_event = receipt.event
        return RelationshipObservationReceipt(
            EventRef(
                source_event.event_id,
                source_event.event_sequence,
                source_event.occurred_at,
            ),
            receipt.interlocutor_key,
            receipt.context_id,
            receipt.context_revision,
            receipt.context_digest,
            receipt.request_digest,
            receipt.before_state_digest,
            receipt.result_state_digest,
            receipt.result_revision,
            receipt.created,
            receipt.previous_receipt_digest,
        )

    @staticmethod
    def _view(root: RelationshipState, key: str) -> RelationshipSelectedView | None:
        record = next(
            (item for item in root.records if item.interlocutor_key == key), None
        )
        if record is None:
            return None
        return RelationshipSelectedView(
            root.revision,
            str(root.canonical_value()["state_digest"]),
            record.relationship_id,
            record.interlocutor_key,
            tuple(
                RelationshipAxisView(
                    axis.kind,
                    axis.interpretation.status,
                    axis.interpretation.estimate,
                    axis.interpretation.confidence,
                    axis.interpretation.accessibility,
                    axis.interpretation.inertia,
                )
                for axis in record.axes
            ),
        )

    def snapshot(self) -> RelationshipState:
        with self._lock:
            return self._copy_root(self._root)

    def _restore_technical_for_agent_state(self, state: RelationshipState) -> None:
        """U5-only exact root restoration; no subjective axis admission."""
        if type(state) is not RelationshipState:
            raise TypeError("Relationship technical state must be exact")
        detached = self._copy_root(state)
        with self._mutation():
            self._root = detached
            self._receipts = ()
            self._last_registry_digest = None

    def selected_view(self, interlocutor_key: str) -> RelationshipSelectedView | None:
        key = identifier(interlocutor_key, "interlocutor_key")
        with self._lock:
            return self._view(self._root, key)

    def _bind_source_qualified_q1_gate(
        self, gate: QualifiedInteractionSourceGate
    ) -> None:
        """One-time existing-Source binding; never a user/model identity setter."""

        if type(gate) is not QualifiedInteractionSourceGate:
            raise TypeError("U2 qualified owner gate must be exact")
        with self._mutation():
            if self._q1_gate is not None:
                raise RelationshipEventConflict("U2 Source gate already bound")
            self._q1_gate = gate

    def _review_source_qualified_q1(self) -> RelationshipIssueReviewResult:
        """Admit exactly one owner-reviewed, fixed Q1 *coordination* hypothesis.

        Difference alone never qualifies. The separate Source gate opens the
        actual committed owners, and a fixed Subject-owned Q1 cooperation rule
        (one mutually selected option) classifies the still-incomplete joint
        coordination. It does NOT identify who the actual speaker is, evaluate
        arbitrary interpersonal conflict or elevate trust.
        """

        gate = self._q1_gate
        if type(gate) is not QualifiedInteractionSourceGate:
            raise QualifiedInteractionUnavailable("U2 Source owner not bound")
        with self._mutation():
            observed = gate.qualify_current_joint_q1()
            witness = observed.experience
            key = observed.interlocutor_key
            event = observed.review_event
            root = self._root
            if root.revision_history:
                last_event = root.revision_history[-1].event
                if (
                    last_event.event_sequence >= event.event_sequence
                    or last_event.occurred_at > event.occurred_at
                ):
                    raise RelationshipEventConflict("U2 review event is stale")
            if root.revision or root.records or root.revision_history:
                if not _source_qualified_q1_root(root, key, witness):
                    raise RelationshipEventConflict(
                        "different Relationship cannot be replaced by Q1"
                    )
                return RelationshipIssueReviewResult(
                    self._copy_root(root), event, False, key
                )
            if (
                len(root.records) >= MAX_RELATIONSHIPS
                or len(root.revision_history) >= MAX_REVISIONS
            ):
                raise RelationshipCapacityExceeded("U2 bounded capacity exceeded")

            # The approved software's one closed joint-Q1 protocol REQUIRES one
            # common final option before completing the exercise. Neither a
            # user's nor a model's "we have a relationship problem" assertion
            # can supply or change this rule. A/B proves only a coordination
            # gap in this scoped exercise, not an objective other-person trait.
            if observed.user_option == observed.self_option:
                raise QualifiedInteractionUnavailable(
                    "joint Q1 has no coordination gap"
                )
            target_id = checksum(
                _COORDINATION_Q1_DOMAIN,
                {"interlocutor_key": key, "experience": witness.reference},
            )
            interpretation = Interpretation(
                target_id,
                InterpretationStatus.PRESENT,
                SCORE_SCALE,
                SCORE_SCALE,
                (witness,),
                (),
                Accessibility.CURRENT,
                Inertia.TENTATIVE,
            )
            unknown = _unknown_record(key, 1)
            axes = tuple(
                RelationshipAxis(axis.kind, interpretation)
                if axis.kind is RelationshipAxisKind.UNRESOLVED_ISSUE
                else axis
                for axis in unknown.axes
            )
            subject_record = RelationshipRecord(
                unknown.relationship_id, key, None, 1, axes
            )
            records = (subject_record,)
            proof = RevisionProof(
                1,
                event,
                target_id,
                _projection_digest(root.revision, root.records),
                _projection_digest(1, records),
                (witness.source_digest,),
                (),
                RevisionReason.INITIAL_INTERPRETATION,
                None,
            )
            candidate = RelationshipState(1, 1, records, None, (proof,))
            if self._reentrant_attempted:
                raise RelationshipEventConflict("reentrant U2 adoption refused")
            if not _source_qualified_q1_root(candidate, key, witness):
                raise RelationshipEventConflict("U2 candidate proof is invalid")
            detached = self._copy_root(candidate)
            # Existing Actor's v10/WAL durability checkpoint owns actual
            # publication and full restore after handler failure.
            self._root = detached
            return RelationshipIssueReviewResult(
                self._copy_root(detached), event, True, key
            )

    def observe_current_interlocutor(
        self, event: EventRef, interlocutor_key: str
    ) -> RelationshipObservationResult | RelationshipHistoricalAcknowledgement:
        """Record R09 opaque membership as UNKNOWN; nothing about trust/identity.

        The future serialized caller must authenticate event and keep R09
        stable. A previous exact operation is replayable only in this instance;
        older/unverifiable retries and any receipt/history overflow fail closed.
        """

        if type(event) is not EventRef:
            raise TypeError("event must be an exact EventRef")
        event.__post_init__()
        # A frozen caller-owned EventRef can still be changed with
        # object.__setattr__; retain only our own validated event copy in
        # revision proofs and receipts so the accepted identity cannot drift.
        event = EventRef(event.event_id, event.event_sequence, event.occurred_at)
        key = identifier(interlocutor_key, "interlocutor_key")
        request_digest = checksum(
            _REQUEST_DOMAIN, {"event": encode(event), "interlocutor_key": key}
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
                        raise RelationshipEventConflict(
                            "event identity or input conflicts with retained proof"
                        )
                    original_receipt = self._copy_receipt(receipt)
                    if (
                        self._root.revision != receipt.result_revision
                        or self._root.canonical_value()["state_digest"]
                        != receipt.result_state_digest
                    ):
                        return RelationshipHistoricalAcknowledgement(original_receipt)
                    snapshot = self._copy_root(self._root)
                    view = self._view(snapshot, key)
                    assert view is not None
                    return RelationshipObservationResult(
                        snapshot, original_receipt, True, view
                    )
            if self._receipts:
                previous_event = self._receipts[-1].event
                if (
                    event.event_sequence <= previous_event.event_sequence
                    or event.occurred_at < previous_event.occurred_at
                ):
                    raise RelationshipEventConflict("event is stale or out of order")
            if len(self._receipts) >= MAX_OBSERVATION_RECEIPTS:
                raise RelationshipCapacityExceeded("observation receipt window is full")

            try:
                state = self._registry.state
                frame = _current_frame(state, event, key)
            except RelationshipSourceUnavailable:
                raise
            except Exception:
                raise RelationshipSourceUnavailable(
                    "R09 current projection could not be read"
                ) from None
            registry_digest = _r09_authority_digest(state)
            if self._receipts:
                latest = self._receipts[-1]
                if state.revision < latest.context_revision or (
                    state.revision == latest.context_revision
                    and registry_digest != self._last_registry_digest
                ):
                    raise RelationshipSourceUnavailable(
                        "R09 source generation regressed or diverged"
                    )
            existing = next(
                (item for item in self._root.records if item.interlocutor_key == key),
                None,
            )
            prior = self._root
            candidate = prior
            if existing is None:
                if (
                    len(prior.records) >= MAX_RELATIONSHIPS
                    or len(prior.revision_history) >= MAX_REVISIONS
                ):
                    raise RelationshipCapacityExceeded(
                        "relationship record or proof window is full"
                    )
                if prior.revision >= MAX_PERSISTED_REVISION:
                    raise RelationshipCapacityExceeded(
                        "relationship revision is exhausted"
                    )
                revision = prior.revision + 1
                new_record = _unknown_record(key, revision)
                records = tuple(
                    sorted(
                        (*prior.records, new_record),
                        key=lambda item: item.relationship_id,
                    )
                )
                proof = RevisionProof(
                    revision,
                    event,
                    new_record.relationship_id,
                    _projection_digest(prior.revision, prior.records),
                    _projection_digest(revision, records),
                    (),
                    (),
                    RevisionReason.INITIAL_INTERPRETATION,
                    prior.revision_history[-1].record_digest
                    if prior.revision_history
                    else None,
                )
                candidate = RelationshipState(
                    1, revision, records, None, prior.revision_history + (proof,)
                )
            # One complete R09 read; reject observable intervening changes to
            # Context or declarative bindings before any process-local write.
            try:
                if self._registry.state != state:
                    raise RelationshipSourceUnavailable(
                        "R09 projection changed during observation"
                    )
            except RelationshipSourceUnavailable:
                raise
            except Exception:
                raise RelationshipSourceUnavailable(
                    "R09 revision recheck failed"
                ) from None
            if self._reentrant_attempted:
                raise RelationshipEventConflict(
                    "reentrant observation invalidated the event"
                )

            snapshot = self._copy_root(candidate)
            view = self._view(snapshot, key)
            assert view is not None
            before_state = str(prior.canonical_value()["state_digest"])
            result_state = str(snapshot.canonical_value()["state_digest"])
            receipt = RelationshipObservationReceipt(
                event,
                key,
                frame.context_id,
                state.revision,
                _r09_digest(state, frame, event),
                request_digest,
                before_state,
                result_state,
                snapshot.revision,
                existing is None,
                self._receipts[-1].receipt_digest if self._receipts else None,
            )
            updated_receipts = self._receipts + (receipt,)
            result = RelationshipObservationResult(
                self._copy_root(snapshot), self._copy_receipt(receipt), False, view
            )
            # Only after every source, root, receipt and read-view preflight is
            # complete do we publish the immutable process-local bundle.
            self._root, self._receipts, self._last_registry_digest = (
                snapshot,
                updated_receipts,
                registry_digest,
            )
            return result
