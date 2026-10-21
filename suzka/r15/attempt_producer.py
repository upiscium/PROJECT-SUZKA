"""R15 U4b: one runtime-bound, source-qualified first-person task attempt.

This read is NOT a verified-competence issuer, self-report trust shortcut,
general task classifier, identity proof, or independent Source/WAL authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from suzka.r15.common import EventRef, SourceWitness
from suzka.r15.continuity_state import ContinuityInvalid
from suzka.r15.private_projection import _committed_root, _qualify_source_meaning
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
)
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import (
    AgentStateLoadError,
    AgentStateSnapshotV10,
    AgentStateStore,
)

if TYPE_CHECKING:
    from suzka.r15.state_port import R15StatePort


TASK_CLASS = "task:referent-resolution"


class QualifiedSelfAttemptUnavailable(ValueError):
    """Bounded failure: no raw source, user text, identity or competence."""


@dataclass(frozen=True, slots=True)
class QualifiedSelfAttempt:
    """Internal observation, never a caller/model-supplied admission warrant."""

    review_event: EventRef
    original_experience: SourceWitness


class QualifiedSelfAttemptGate:
    """Revalidate one previously U3-admitted attempt in a *real* Actor event.

    The bound existing owner stores, not an operator-provided candidate, supply
    the U3 source refs and actual committed R07/R12/DB1 evidence.
    """

    def __init__(
        self,
        store: AgentStateStore,
        port: R15StatePort,
        source_gate: QualifiedNarrativeSourceGate,
        runtime: AgentRuntime,
    ) -> None:
        from suzka.r15.state_port import R15StatePort

        if (
            type(store) is not AgentStateStore
            or type(port) is not R15StatePort
            or type(source_gate) is not QualifiedNarrativeSourceGate
            or type(runtime) is not AgentRuntime
            or source_gate._runtime is not runtime
            or port.narrative._source_gate is not source_gate
        ):
            raise QualifiedSelfAttemptUnavailable("SelfModel Source owners unbound")
        self._store = store
        self._port = port
        self._source_gate = source_gate
        self._runtime = runtime

    def qualify_current_attempt(self) -> QualifiedSelfAttempt:
        event: AgentEvent | None = self._runtime.current_event()
        if (
            type(event) is not AgentEvent
            or event.event_type is not AgentEventType.SELF_ATTEMPT_REVIEW
            or event.source is not AgentEventSource.RUNTIME_SELF_ATTEMPT_REVIEW
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
            or self._port.narrative._source_gate is not self._source_gate
            or self._port._private_read_store is not self._store
            or self._port._private_read_runtime is not self._runtime
        ):
            raise QualifiedSelfAttemptUnavailable(
                "SelfModel attempt requires the internal serialized owner event"
            )
        try:
            before = self._store.load()
            if not isinstance(before, AgentStateSnapshotV10):
                raise QualifiedSelfAttemptUnavailable(
                    "committed v10 SelfModel evidence unavailable"
                )
            root, continuation = _committed_root(before, self._port)
            owned = self._port._capture_for_agent_state()
            if owned.self_model.canonical_value() != before.self_model_state:
                raise QualifiedSelfAttemptUnavailable(
                    "SelfModel owner differs from committed AgentState"
                )
            if continuation.context_id is None:
                raise QualifiedSelfAttemptUnavailable("U3 attempt context unavailable")

            # Both S1-only and S2-reinterpreted genuine U3 histories qualify.
            # This reconstructs first-person A/B from actual owner records,
            # rather than treating a typed model answer as a provenance issuer.
            _qualify_source_meaning(
                root,
                continuation,
                self._source_gate,
                context_id=continuation.context_id,
                event_sequence=event.processing_sequence,
            )
            source = self._source_gate._read(continuation.first_sources[1])
            if (
                len(tuple(ep for ep in root.episodes if not ep.links)) != 1
                or source.witness
                != next(ep for ep in root.episodes if not ep.links).experience
                or source.event.event_sequence >= event.processing_sequence
                or source.event.occurred_at > event.requested_at
            ):
                raise QualifiedSelfAttemptUnavailable(
                    "observed first-person attempt is not source-backed"
                )

            after = self._store.load()
            if not isinstance(after, AgentStateSnapshotV10) or after != before:
                raise QualifiedSelfAttemptUnavailable(
                    "committed SelfModel state changed during review"
                )
            _committed_root(after, self._port)
            if (
                self._port._capture_for_agent_state().self_model.canonical_value()
                != after.self_model_state
            ):
                raise QualifiedSelfAttemptUnavailable(
                    "attempt owner changed during source read"
                )
            return QualifiedSelfAttempt(
                EventRef(event.event_id, event.processing_sequence, event.requested_at),
                source.witness,
            )
        except (
            AgentStateLoadError,
            ContinuityInvalid,
            QualifiedProposalUnavailable,
            QualifiedSelfAttemptUnavailable,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            RecursionError,
        ) as exc:
            raise QualifiedSelfAttemptUnavailable(
                "source-qualified SelfModel attempt unavailable"
            ) from exc
