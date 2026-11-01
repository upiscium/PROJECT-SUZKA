"""Private U2a source-only producer, not R16 Plan or Goal admission.

Owns only ephemeral presentation/revalidation in the single serialized app
runtime. R07, DB1, R09 and R12 remain source authorities; R13 remains mutation
authority and existing AgentState v10 ports own durability. No text, selected
Goal, caller appraisal, model invocation or public route is accepted here.
This cooperative process-local boundary is not a Python sandbox or an external
filesystem-tampering fence.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from suzka.experience import (
    ExperienceLifecycle,
    ExperienceRecord,
    experience_record_digest,
)
from suzka.identifiers import validate_identifier
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE
from suzka.memory import DualMemorySystem, ExperienceStore
from suzka.memory.dual_memory_system import (
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
)
from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.experience_store import ExperienceStoreError
from suzka.memory.memory_schema import MemoryRecordType
from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.system import MotivationEvidence, MotivationExperienceEvidence
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.context import (
    ContextNotFound,
    ContextRegistry,
    ContextStateInvalid,
    ContextStatus,
)
from suzka.runtime.event_journal import EventJournal, EventJournalError, EventLifecycle

if TYPE_CHECKING:
    from suzka.runtime.main_loop import SuzkaMainLoop


class _MotivationSourceUnavailable(ValueError):
    """Coarse private refusal; never includes source text or metadata."""


@dataclass(frozen=True, slots=True)
class _MotivationSourceRef:
    """Selection only, NOT evidence or proof of source authority."""

    event_id: str
    event_sequence: int
    episode_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.event_id)
        validate_identifier(self.episode_id)
        if (
            type(self.event_sequence) is not int
            or not 1 <= self.event_sequence <= MAX_PERSISTED_EVENT_SEQUENCE
        ):
            raise _MotivationSourceUnavailable("invalid source order")


@dataclass(frozen=True, slots=True)
class _PendingSource:
    event: AgentEvent
    source: _MotivationSourceRef
    context_id: str
    source_channel: str
    source_session_id: str | None
    experience_id: str
    revision: int
    record_digest: str


class _MotivationSourceOwner:
    """App-bound reader + nonce-bound presenter; no evidence-taking entrypoint."""

    def __init__(
        self,
        loop: SuzkaMainLoop,
        runtime: AgentRuntime,
        journal: EventJournal,
        experience: ExperienceStore,
        memory: DualMemorySystem,
        contexts: ContextRegistry,
    ) -> None:
        from suzka.runtime.main_loop import SuzkaMainLoop

        if (
            not isinstance(loop, SuzkaMainLoop)
            or type(runtime) is not AgentRuntime
            or type(journal) is not EventJournal
            or type(experience) is not ExperienceStore
            or type(memory) is not DualMemorySystem
            or type(contexts) is not ContextRegistry
            or not journal.has_exclusive_authority
            or loop._runtime is not runtime
            or loop.experience_store is not experience
            or loop.memory_system is not memory
            or loop.context_registry is not contexts
            or experience.root.resolve()
            != ExperienceStore.from_memory_root(
                memory.settings.memory.persist_directory
            ).root.resolve()
        ):
            raise _MotivationSourceUnavailable("bound source owners unavailable")
        self._loop = loop
        self._runtime = runtime
        self._journal = journal
        self._experience = experience
        self._memory = memory
        self._contexts = contexts
        self._inflight: tuple[object, _MotivationSourceRef] | None = None
        self._pending: _PendingSource | None = None
        loop._bind_motivation_source_owner(self)

    def _check_binding(self) -> None:
        if (
            self._loop._runtime is not self._runtime
            or self._loop.experience_store is not self._experience
            or self._loop.memory_system is not self._memory
            or self._loop.context_registry is not self._contexts
            or self._loop._motivation_source_owner is not self
        ):
            raise _MotivationSourceUnavailable("source owner binding changed")

    def _submit(self, source: _MotivationSourceRef) -> Future[AgentEventOutcome[str]]:
        # Exact closed shape excludes model/operator content and extra hashes.
        if type(source) is not _MotivationSourceRef:
            raise _MotivationSourceUnavailable("untrusted source selection shape")
        source.__post_init__()
        self._check_binding()
        nonce = object()

        def handle() -> str:
            if self._inflight is not None or self._pending is not None:
                raise _MotivationSourceUnavailable(
                    "source presentation already pending"
                )
            self._inflight = (nonce, source)
            try:
                return self._loop._apply_motivation_source(self, nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(
            AgentEventType.MOTIVATION_SOURCE_REVIEW,
            AgentEventSource.RUNTIME_MOTIVATION_SOURCE_REVIEW,
            handle,
        )

    def _present(
        self, loop: SuzkaMainLoop, nonce: object
    ) -> tuple[AgentEvent, MotivationEvidence | None]:
        self._check_binding()
        event = self._runtime.current_event()
        if (
            loop is not self._loop
            or self._inflight is None
            or self._inflight[0] is not nonce
            or self._pending is not None
            or event is None
            or event.event_type is not AgentEventType.MOTIVATION_SOURCE_REVIEW
            or event.source is not AgentEventSource.RUNTIME_MOTIVATION_SOURCE_REVIEW
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _MotivationSourceUnavailable("not a bound worker presentation")
        record, pending = self._read(self._inflight[1], event)
        self._pending = pending
        appraisal = record.appraisal
        if (
            not appraisal.novelty_valid
            and appraisal.goal_progress is None
            and appraisal.threat is None
        ):
            # Lingering negative emotion alone cannot invent a missing appraisal.
            return event, None
        evidence = MotivationEvidence.from_experience(
            MotivationExperienceEvidence(
                experience_ref=R13Reference(
                    R13ReferenceKind.EXPERIENCE, record.experience_id
                ),
                source_event_id=record.source_event_id,
                source_event_sequence=record.source_event_sequence,
                created_at=record.created_at,
                active=True,
                novelty=appraisal.novelty,
                novelty_valid=appraisal.novelty_valid,
                goal_progress=appraisal.goal_progress,
                threat=appraisal.threat,
                emotion_valence=record.post_appraisal_emotion.valence,
                subjective_salience=record.subjective_salience,
            )
        )
        return event, (
            None
            if evidence is None
            else replace(
                evidence,
                authority_witness_digest=pending.record_digest,
            )
        )

    def _read(
        self, source: _MotivationSourceRef, event: AgentEvent
    ) -> tuple[ExperienceRecord, _PendingSource]:
        self._check_binding()
        sequence = event.processing_sequence
        if type(sequence) is not int or source.event_sequence >= sequence:
            raise _MotivationSourceUnavailable("source must precede review")
        try:
            inspected = self._journal.inspect()
            if (
                inspected.external_reconciliation_required
                or inspected.open_transactions
                or inspected.reconciliation_required_transactions
            ):
                raise _MotivationSourceUnavailable("Journal source is in recovery")
            relevant = tuple(
                row for row in inspected.records if row.event_id == source.event_id
            )
            for lifecycle in (
                EventLifecycle.ACCEPTED,
                EventLifecycle.COMPLETED,
                EventLifecycle.TRANSACTION_COMPLETED,
            ):
                rows = tuple(row for row in relevant if row.lifecycle is lifecycle)
                if (
                    len(rows) != 1
                    or rows[0].event_type is not AgentEventType.CHAT
                    or rows[0].source is not AgentEventSource.API_CHAT
                    or (
                        lifecycle is not EventLifecycle.ACCEPTED
                        and rows[0].processing_sequence != source.event_sequence
                    )
                    or rows[0].timestamp > event.requested_at
                ):
                    raise _MotivationSourceUnavailable("R07 source not terminal CHAT")
            if any(row.lifecycle is EventLifecycle.FAILED for row in relevant):
                raise _MotivationSourceUnavailable("R07 source failed")
            experience_id = experience_id_for_event(
                source.event_id, source.event_sequence
            )
            record = self._experience.load_committed_current(experience_id).record
            if (
                record.lifecycle is not ExperienceLifecycle.ACTIVE
                or record.experience_id != experience_id
                or record.source_event_id != source.event_id
                or record.source_event_sequence != source.event_sequence
                or record.source_episode_id != source.episode_id
                or record.created_at > event.requested_at
            ):
                raise _MotivationSourceUnavailable("R12 source inconsistent")
            committed = self._memory.get_committed_episodic(source.episode_id)
            if committed is None:
                raise _MotivationSourceUnavailable("DB1 source not committed")
            episode = committed.record
            context = self._contexts.get(record.context_id)
            if (
                episode.id != source.episode_id
                or episode.record_type is not MemoryRecordType.EPISODIC_LOG
                or episode.archived
                or episode.context_id != record.context_id
                or context.context_id != record.context_id
                or context.status is not ContextStatus.ACTIVE
                or episode.source_channel != context.source_channel
                or episode.source_session_id != context.source_session_id
                or context.participant_refs
            ):
                # No event-time participant identity is admitted in this scope.
                raise _MotivationSourceUnavailable("DB1/Context source ambiguous")
            return record, _PendingSource(
                event,
                source,
                record.context_id,
                context.source_channel,
                context.source_session_id,
                experience_id,
                record.revision,
                experience_record_digest(record),
            )
        except (
            EventJournalError,
            ExperienceStoreError,
            EpisodicMemoryReadError,
            EpisodicMemoryFormatError,
            ContextNotFound,
            ContextStateInvalid,
            ValueError,
            OSError,
        ) as error:
            if isinstance(error, _MotivationSourceUnavailable):
                raise
            raise _MotivationSourceUnavailable(
                "source owner read unavailable"
            ) from error

    def _validate_commit(self, event: AgentEvent) -> None:
        # current_event() intentionally ends with the handler. Here the app's
        # checkpoint must match the exact event saved by the worker presenter.
        pending = self._pending
        if event.event_type is not AgentEventType.MOTIVATION_SOURCE_REVIEW:
            if pending is not None:
                raise _MotivationSourceUnavailable("pending source event mismatch")
            return
        if pending is None or pending.event != event:
            raise _MotivationSourceUnavailable("source presentation unavailable")
        _, current = self._read(pending.source, event)
        if current != pending:
            raise _MotivationSourceUnavailable("source changed before WAL capture")

    def _clear(self, event: AgentEvent) -> None:
        del event
        self._pending = None
        self._inflight = None
