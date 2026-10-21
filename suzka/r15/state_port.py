"""One U5-only R15 technical state port; domain owners retain mutation authority.

The port exports/restores exact U1 roots for the existing *single process*
and is never a public NarrativeSelf trust/trait/capability writer.
WAL/event atomicity remains solely AgentStateStore/StateRecovery's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from suzka.r15.interaction_producer import (
    QualifiedInteractionSourceGate,
    QualifiedInteractionUnavailable,
)
from suzka.r15.attempt_producer import (
    QualifiedSelfAttemptGate,
    QualifiedSelfAttemptUnavailable,
)
from suzka.r15.private_projection import (
    HOLD,
    PrivateNarrativeProjection,
    read_private_committed_narrative,
)
from suzka.r15.qualified_proposal import QualifiedNarrativeSourceGate
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateStore
from suzka.runtime.event_journal import EventJournal
from suzka.runtime.state_wal import StateWAL
from suzka.memory import DualMemorySystem, ExperienceStore

from suzka.r15.continuity_state import NarrativeContinuation
from suzka.r15.contracts import (
    NarrativeSelfState,
    RelationshipState,
    SelfModelState,
)
from suzka.r15.narrative_self_system import NarrativeSelfSystem
from suzka.r15.relationship_system import (
    RelationshipIssueReviewResult,
    RelationshipSystem,
)
from suzka.r15.self_model_system import SelfAttemptResult, SelfModelSystem
from suzka.runtime.context import ContextRegistry


@dataclass(frozen=True, slots=True)
class R15TechnicalSnapshot:
    relationship: RelationshipState
    narrative: NarrativeSelfState
    self_model: SelfModelState
    continuation: NarrativeContinuation

    def revalidate(self) -> R15TechnicalSnapshot:
        if (
            type(self.relationship) is not RelationshipState
            or type(self.narrative) is not NarrativeSelfState
            or type(self.self_model) is not SelfModelState
            or type(self.continuation) is not NarrativeContinuation
        ):
            raise ValueError("R15 technical snapshot has invalid domain owners")
        rel = RelationshipState.from_json(self.relationship.canonical_bytes())
        narrative = NarrativeSelfState.from_json(self.narrative.canonical_bytes())
        self_model = SelfModelState.from_json(self.self_model.canonical_bytes())
        continuation = NarrativeContinuation.from_value(
            self.continuation.canonical_value(narrative), narrative
        )
        return R15TechnicalSnapshot(rel, narrative, self_model, continuation)

    @classmethod
    def empty(cls) -> R15TechnicalSnapshot:
        relationship = RelationshipState(1, 0, (), None, ())
        narrative = NarrativeSelfState(1, 0, (), (), None, ())
        self_model = SelfModelState(1, 0, (), (), None, ())
        return cls(
            relationship, narrative, self_model, NarrativeContinuation.empty(narrative)
        )


class R15StatePort:
    """A private U5 delegation port for exactly one U2/U3/U4 owner group."""

    def __init__(self, context: ContextRegistry) -> None:
        if type(context) is not ContextRegistry:
            raise TypeError("R15 owner port requires an exact R09 ContextRegistry")
        self._owner_lock = RLock()
        self.relationship = RelationshipSystem(context)
        self.narrative = NarrativeSelfSystem()
        self.self_model = SelfModelSystem()
        self._private_read_store: AgentStateStore | None = None
        self._private_read_runtime: AgentRuntime | None = None

    def _bind_private_read_sources(
        self, store: AgentStateStore, runtime: AgentRuntime
    ) -> None:
        """Startup-only binding to the EXISTING WAL and serialized runtime.

        The qualified gate is already the one bound to the Narrative owner.
        No second StateStore, public API, operator/model override or setter.
        """
        if not isinstance(store, AgentStateStore) or type(runtime) is not AgentRuntime:
            raise TypeError("private read requires exact existing owners")
        with self._owner_lock:
            gate = self.narrative._source_gate
            if (
                type(gate) is not QualifiedNarrativeSourceGate
                or gate._runtime is not runtime
            ):
                raise RuntimeError("private read source owners are not bound")
            if self._private_read_store is not None:
                if (
                    self._private_read_store is store
                    and self._private_read_runtime is runtime
                ):
                    return
                raise RuntimeError("private read owners cannot be rebound")
            # Existing AgentStateStore subclasses may be injected by older
            # recovery tests. Only the exact production owner can authorize
            # a new positive U4; subclasses retain the old fail-closed path.
            if type(store) is AgentStateStore:
                self.self_model._bind_source_qualified_attempt_gate(
                    QualifiedSelfAttemptGate(store, self, gate, runtime)
                )
            self._private_read_store = store
            self._private_read_runtime = runtime

    def _review_source_qualified_self_attempt(self) -> SelfAttemptResult:
        """Internal self-attempt adoption, never callable as a model setter."""

        with self._owner_lock:
            runtime = self._private_read_runtime
            event = None if runtime is None else runtime.current_event()
            if (
                type(runtime) is not AgentRuntime
                or type(event) is not AgentEvent
                or event.event_type is not AgentEventType.SELF_ATTEMPT_REVIEW
                or event.source is not AgentEventSource.RUNTIME_SELF_ATTEMPT_REVIEW
                or type(event.processing_sequence) is not int
                or event.processing_sequence < 1
            ):
                raise QualifiedSelfAttemptUnavailable(
                    "SelfModel revision requires the internal Actor event"
                )
            return self.self_model._review_source_qualified_attempt()

    def _bind_private_relationship_sources(
        self,
        store: AgentStateStore,
        runtime: AgentRuntime,
        journal: EventJournal,
        experience: ExperienceStore,
        memory: DualMemorySystem,
        wal: StateWAL,
        contexts: ContextRegistry,
    ) -> None:
        """One existing-owner binding, no R09/R12/StateWAL writer or API."""

        with self._owner_lock:
            if (
                type(store) is not AgentStateStore
                or type(runtime) is not AgentRuntime
                or type(journal) is not EventJournal
                or type(experience) is not ExperienceStore
                or type(memory) is not DualMemorySystem
                or type(wal) is not StateWAL
                or contexts is not self.relationship._registry
                or store is not self._private_read_store
                or runtime is not self._private_read_runtime
            ):
                raise QualifiedInteractionUnavailable("U2 source owners mismatch")
            self.relationship._bind_source_qualified_q1_gate(
                QualifiedInteractionSourceGate(
                    store,
                    self,
                    runtime,
                    journal,
                    experience,
                    memory,
                    wal,
                    contexts,
                )
            )

    def _review_source_qualified_q1_interaction(self) -> RelationshipIssueReviewResult:
        """Only an actual serial RELATIONSHIP_REVIEW may mutate U2."""

        with self._owner_lock:
            runtime = self._private_read_runtime
            event = None if runtime is None else runtime.current_event()
            if (
                type(runtime) is not AgentRuntime
                or type(event) is not AgentEvent
                or event.event_type is not AgentEventType.RELATIONSHIP_REVIEW
                or event.source is not AgentEventSource.RUNTIME_RELATIONSHIP_REVIEW
                or type(event.processing_sequence) is not int
                or event.processing_sequence < 1
            ):
                raise QualifiedInteractionUnavailable(
                    "U2 admission requires actual internal Actor event"
                )
            return self.relationship._review_source_qualified_q1()

    def _read_private_narrative_in_chat(
        self, event: AgentEvent, context_id: str
    ) -> PrivateNarrativeProjection:
        """An internal read ONLY inside the actual CHAT/API_CHAT worker."""
        with self._owner_lock:
            runtime = self._private_read_runtime
            store = self._private_read_store
            gate = self.narrative._source_gate
            if (
                type(event) is not AgentEvent
                or runtime is None
                or store is None
                or type(gate) is not QualifiedNarrativeSourceGate
                or gate._runtime is not runtime
                or runtime.current_event() is not event
                or event.event_type is not AgentEventType.CHAT
                or event.source is not AgentEventSource.API_CHAT
                or type(event.processing_sequence) is not int
                or type(context_id) is not str
            ):
                return HOLD
            return read_private_committed_narrative(
                store,
                self,
                gate,
                context_id=context_id,
                event_sequence=event.processing_sequence,
            )

    def _capture_for_agent_state(self) -> R15TechnicalSnapshot:
        """Detach all three roots, U3 evidence scope and review under one port."""
        with self._owner_lock:
            root, continuation = self.narrative._technical_state_for_agent_state()
            return R15TechnicalSnapshot(
                self.relationship.snapshot(),
                root,
                self.self_model.snapshot(),
                continuation,
            ).revalidate()

    def _restore_for_agent_state(self, state: R15TechnicalSnapshot) -> None:
        """U5-only exact restore with best-effort rollback and fail-closed errors."""
        if type(state) is not R15TechnicalSnapshot:
            raise TypeError("R15 restore requires an exact technical snapshot")
        checked = state.revalidate()
        with self._owner_lock:
            original = self._capture_for_agent_state()
            try:
                self.relationship._restore_technical_for_agent_state(
                    checked.relationship
                )
                self.self_model._restore_technical_for_agent_state(checked.self_model)
                self.narrative._restore_technical_for_agent_state(
                    checked.narrative, checked.continuation
                )
                if self._capture_for_agent_state() != checked:
                    raise RuntimeError("R15 technical restore did not publish exactly")
            except Exception as exc:
                # These are internal technical restores, not independent
                # Source admissions. Rollback failures must be observable and
                # cause a runtime fail-stop; never return a partial success.
                try:
                    self.relationship._restore_technical_for_agent_state(
                        original.relationship
                    )
                    self.self_model._restore_technical_for_agent_state(
                        original.self_model
                    )
                    self.narrative._restore_technical_for_agent_state(
                        original.narrative, original.continuation
                    )
                except Exception as rollback_error:
                    raise RuntimeError(
                        "R15 technical restore and rollback both failed"
                    ) from rollback_error
                raise RuntimeError("R15 technical restore refused") from exc
