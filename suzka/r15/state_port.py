"""One U5-only R15 technical state port; domain owners retain mutation authority.

The port exports/restores exact U1 roots for the existing *single process*
and is never a public NarrativeSelf trust/trait/capability writer.
WAL/event atomicity remains solely AgentStateStore/StateRecovery's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from suzka.r15.continuity_state import NarrativeContinuation
from suzka.r15.contracts import (
    NarrativeSelfState,
    RelationshipState,
    SelfModelState,
)
from suzka.r15.narrative_self_system import NarrativeSelfSystem
from suzka.r15.relationship_system import RelationshipSystem
from suzka.r15.self_model_system import SelfModelSystem
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
