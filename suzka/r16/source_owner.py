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
from enum import Enum
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
from suzka.motivation.system import (
    MotivationEvidence,
    MotivationExperienceEvidence,
    MotivationMutationEvidence,
)
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
from suzka.runtime.journal_source import (
    JournalSourceReason,
    JournalSourceStage,
    SourceUnavailable,
    locate_journal_source,
)

if TYPE_CHECKING:
    from suzka.runtime.main_loop import SuzkaMainLoop


# Bound on probing committed Experience ids for an archived request (IP-4c).
_ARCHIVED_SEQUENCE_PROBE_LIMIT = 65_536


class _MotivationSourceUnavailable(SourceUnavailable):
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

    def _apply_verified_source(self, loop: SuzkaMainLoop, nonce: object) -> str:
        """Apply one R16-qualified source through existing R13 policy only.

        The worker/event/nonce gate resides in _present; no caller may supply
        evidence or a digest. Existing R13 MotivationSystem remains the
        mutation authority, while MainLoop exposes no direct R13 producer call.
        """
        event, evidence = self._present(loop, nonce)
        if evidence is None:
            return "neutral_no_eligible_appraisal"
        motivation = loop._motivation_system
        if any(
            item.evidence.source_ref == evidence.source_ref
            for item in motivation.snapshot().evidence_ledger
        ):
            raise ValueError("Motivation source identity already consumed")
        assert event.processing_sequence is not None
        motivation.apply_evidence(
            evidence,
            MotivationMutationEvidence(
                event.event_id,
                event.processing_sequence,
                event.requested_at,
                (evidence.source_ref,),
            ),
        )
        return "motivation_evidence_applied"

    def _read(
        self,
        source: _MotivationSourceRef,
        event: AgentEvent,
        *,
        continuation: bool = False,
    ) -> tuple[ExperienceRecord, _PendingSource]:
        """Read one CHAT source through R07, R12, DB1 and Context.

        IP-4c: when the CHAT event is no longer retained, ``continuation``
        decides. Admission (False) refuses it as ``SOURCE_PAST_RETENTION``
        when the committed Experience proves it existed and as
        ``SOURCE_MISSING`` otherwise. Continuation (True) accepts it on the
        committed Experience, which was checked against R07 at admission.
        """
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
            stage, relevant = locate_journal_source(
                inspected, source.event_id, source.event_sequence
            )
            if stage is JournalSourceStage.MISSING:
                raise _MotivationSourceUnavailable(
                    "R07 source missing", reason=JournalSourceReason.SOURCE_MISSING
                )
            for lifecycle in (
                EventLifecycle.ACCEPTED,
                EventLifecycle.COMPLETED,
                EventLifecycle.TRANSACTION_COMPLETED,
            ):
                if stage is not JournalSourceStage.RETAINED:
                    break
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
            try:
                record = self._experience.load_committed_current(experience_id).record
            except ExperienceStoreError as error:
                if stage is JournalSourceStage.RETAINED:
                    raise
                raise _MotivationSourceUnavailable(
                    "archived source unproven",
                    reason=JournalSourceReason.SOURCE_MISSING,
                ) from error
            if stage is JournalSourceStage.ARCHIVED and (
                record.source_event_id != source.event_id
                or record.source_event_sequence != source.event_sequence
            ):
                raise _MotivationSourceUnavailable(
                    "archived source unproven",
                    reason=JournalSourceReason.SOURCE_MISSING,
                )
            if stage is JournalSourceStage.ARCHIVED and not continuation:
                # A new source must be retained: past the window it cannot
                # become the origin of interest or a Goal (#381 §4).
                raise _MotivationSourceUnavailable(
                    "R07 source past retention",
                    reason=JournalSourceReason.SOURCE_PAST_RETENTION,
                )
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

    def _archived_source_sequence(
        self, event_id: str, after: int, before: int
    ) -> int:
        """Recover an archived CHAT's sequence from the committed Experience.

        Continuation only (IP-4c). Owners keep the request event's id but not
        its sequence; once the event is archived, the sequence is found by
        probing ``experience_id_for_event(event_id, s)`` for ``after < s <
        before``, bounds taken from committed owner state. The Experience it
        finds names the event and sequence itself, and event ids are unique
        in the Journal, so a probe can only succeed for the real source.

        The probe runs downward from ``before``: the request is the turn that
        prompted the owner event, so it normally sits just below it. It is
        bounded; past the bound the source is treated as missing (fail
        closed).
        """
        if type(after) is not int or type(before) is not int or after < 0:
            raise _MotivationSourceUnavailable(
                "archived source sequence unbounded",
                reason=JournalSourceReason.SOURCE_MISSING,
            )
        lowest = max(after + 1, before - _ARCHIVED_SEQUENCE_PROBE_LIMIT)
        for candidate in range(before - 1, lowest - 1, -1):
            experience_id = experience_id_for_event(event_id, candidate)
            try:
                if self._experience.load_current(experience_id) is None:
                    continue
                record = self._experience.load_committed_current(experience_id).record
            except ExperienceStoreError:
                # Damaged candidates are skipped; the real source, if
                # damaged, is then missing (fail closed).
                continue
            if (
                record.source_event_id == event_id
                and record.source_event_sequence == candidate
            ):
                return candidate
        raise _MotivationSourceUnavailable(
            "archived source unproven", reason=JournalSourceReason.SOURCE_MISSING
        )

    def _scoped_target_reason(
        self,
        error: _MotivationSourceUnavailable,
        target: _MotivationSourceRef,
        request: _MotivationSourceRef,
        event: AgentEvent,
        *,
        continuation: bool,
    ) -> Enum | None:
        """Release a target's retention reason only inside the request's Context.

        IP-5 review P1 (#370 D2 P2-2): a requester names the target X, so a
        reason derived from X alone (missing, past retention) would reveal
        whether X exists in another Context. It is kept only when the request
        P reads successfully and X's committed Experience exists in P's
        Context; otherwise the refusal is the uniform SOURCE_UNAVAILABLE
        (``None``). A nonexistent X is therefore indistinguishable from an X
        in another Context.
        """
        if error.reason is None:
            return None
        try:
            _request_record, request_source = self._read(
                request, event, continuation=continuation
            )
            target_record = self._experience.load_committed_current(
                experience_id_for_event(target.event_id, target.event_sequence)
            ).record
        except (_MotivationSourceUnavailable, ExperienceStoreError, ValueError, OSError):
            return None
        if (
            target_record.source_event_id != target.event_id
            or target_record.source_event_sequence != target.event_sequence
            or target_record.context_id != request_source.context_id
        ):
            return None
        return error.reason

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
