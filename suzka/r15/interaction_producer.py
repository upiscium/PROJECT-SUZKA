"""R15 U2b: one source-qualified, first-binding-only Q1 interaction observation.

An *existing* Chat may visibly express differing provisional Q1 choices.
This never authenticates an actual person, a speaker on later events, trust,
or a model-supplied verdict that a relationship problem exists.

Only the existing serialized Actor and R15 Subject port can call this gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import TYPE_CHECKING

from suzka.experience import ExperienceLifecycle, experience_record_digest
from suzka.memory import DualMemorySystem, ExperienceStore
from suzka.memory.dual_memory_system import (
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
)
from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.experience_store import ExperienceStoreError
from suzka.memory.memory_schema import MemoryRecordType
from suzka.r15.common import (
    ClaimedOrigin,
    EventRef,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
    canonical_json,
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
from suzka.runtime.context import ContextRegistry, ContextStatus
from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalError,
    EventLifecycle,
)
from suzka.runtime.state_wal import (
    BaselineRecord,
    StateWAL,
    StateWALError,
    TransitionRecord,
)

if TYPE_CHECKING:
    from suzka.r15.state_port import R15StatePort


# This is a CLOSED, implementation-owned Q1 cooperation exercise. The rule is
# "one common choice before joint completion"; it is not read from a caller
# statement or generated answer, nor is it a general Subject/Goal service.
_Q1_USER = re.compile(
    r"共同の比較課題Q1について私は案([AB])を暫定的に選びます。"
    r"(?:結論はまだ未決定で保留中です。)?"
)
_Q1_SELF = re.compile(
    r"共同の比較課題Q1についてSuzkaは案([AB])を暫定的に選びます。"
    r"(?:結論は未決定のため保留します。)?正解は未検証です。"
)
_MAX_MESSAGE_BYTES = 4096


class QualifiedInteractionUnavailable(ValueError):
    """Coarse refusal; never disclose private user/model Source text."""


@dataclass(frozen=True, slots=True)
class QualifiedQ1Coordination:
    """One verified event lineage; pending *exercise* not a person's intent."""

    review_event: EventRef
    observed_event: EventRef
    experience: SourceWitness
    interlocutor_key: str
    context_id: str
    user_option: str
    self_option: str

    def __post_init__(self) -> None:
        if (
            type(self.review_event) is not EventRef
            or type(self.observed_event) is not EventRef
            or type(self.experience) is not SourceWitness
            or type(self.interlocutor_key) is not str
            or type(self.context_id) is not str
            or self.user_option not in ("A", "B")
            or self.self_option not in ("A", "B")
            or self.user_option == self.self_option
            or self.observed_event.event_sequence >= self.review_event.event_sequence
            or self.observed_event.occurred_at > self.review_event.occurred_at
        ):
            raise QualifiedInteractionUnavailable("unqualified Q1 interaction")


class QualifiedInteractionSourceGate:
    """Reopen the actual R07/R12/DB1/WAL owners under one Actor review.

    Candidate selection is deterministic and has ZERO caller/model inputs.
    The only semantic rule is the fixed, code-owned joint-Q1 consensus
    criterion. Whether to adopt the observed gap belongs to RelationshipSystem.
    """

    def __init__(
        self,
        store: AgentStateStore,
        port: R15StatePort,
        runtime: AgentRuntime,
        journal: EventJournal,
        experience: ExperienceStore,
        memory: DualMemorySystem,
        wal: StateWAL,
        contexts: ContextRegistry,
    ) -> None:
        from suzka.r15.state_port import R15StatePort

        if (
            type(store) is not AgentStateStore
            or type(port) is not R15StatePort
            or type(runtime) is not AgentRuntime
            or type(journal) is not EventJournal
            or type(experience) is not ExperienceStore
            or type(memory) is not DualMemorySystem
            or type(wal) is not StateWAL
            or type(contexts) is not ContextRegistry
            or port.relationship._registry is not contexts
            or port._private_read_runtime is not runtime
            or port._private_read_store is not store
        ):
            raise QualifiedInteractionUnavailable("U2 Source owners unbound")
        self._store = store
        self._port = port
        self._runtime = runtime
        self._journal = journal
        self._experience = experience
        self._memory = memory
        self._wal = wal
        self._contexts = contexts

    def qualify_current_joint_q1(self) -> QualifiedQ1Coordination:
        event: AgentEvent | None = self._runtime.current_event()
        if (
            type(event) is not AgentEvent
            or event.event_type is not AgentEventType.RELATIONSHIP_REVIEW
            or event.source is not AgentEventSource.RUNTIME_RELATIONSHIP_REVIEW
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
            or self._port._private_read_store is not self._store
            or self._port._private_read_runtime is not self._runtime
            or self._port.relationship._registry is not self._contexts
        ):
            raise QualifiedInteractionUnavailable(
                "U2 positive review requires the internal serialized Actor event"
            )

        try:
            before = self._store.load()
            if not isinstance(before, AgentStateSnapshotV10):
                raise QualifiedInteractionUnavailable("committed v10 unavailable")
            if (
                self._port._capture_for_agent_state().relationship.canonical_value()
                != before.relationship_state
            ):
                raise QualifiedInteractionUnavailable(
                    "Relationship owner differs from committed v10"
                )

            inspected = self._journal.inspect()
            if (
                inspected.external_reconciliation_required
                or inspected.open_transactions
                or inspected.reconciliation_required_transactions
            ):
                raise QualifiedInteractionUnavailable("R07 source is in recovery")

            # Only the *latest* actually completed ordinary Chat in this
            # journal can be considered. Never hunt an older, more favorable
            # Q1 message after a later Chat has already happened.
            completed = tuple(
                row
                for row in inspected.records
                if row.lifecycle is EventLifecycle.COMPLETED
                and row.event_type is AgentEventType.CHAT
                and row.source is AgentEventSource.API_CHAT
            )
            if not completed:
                raise QualifiedInteractionUnavailable("recent Chat source unavailable")
            terminal = completed[-1]
            eid = terminal.event_id
            sequence = terminal.processing_sequence
            if (
                type(eid) is not str
                or type(sequence) is not int
                or sequence <= 0
                or sequence >= event.processing_sequence
            ):
                raise QualifiedInteractionUnavailable("recent event identity invalid")
            # Even a later failed/uncommitted Chat may have temporarily
            # changed a request's Context before rollback. Avoid pronouncing
            # an older Q1 current when such a more recent dialogue event
            # exists; neither current Context membership nor rollback text
            # can prove its speaker or resolution semantics.
            if any(
                row.event_type in (AgentEventType.CHAT, AgentEventType.DEBUG_CHAT)
                and type(row.processing_sequence) is int
                and row.processing_sequence > sequence
                and row.lifecycle
                in (
                    EventLifecycle.ACCEPTED,
                    EventLifecycle.STARTED,
                    EventLifecycle.COMPLETED,
                    EventLifecycle.FAILED,
                )
                for row in inspected.records
            ):
                raise QualifiedInteractionUnavailable(
                    "a later dialogue event makes Q1 attribution stale"
                )
            related = tuple(row for row in inspected.records if row.event_id == eid)
            accepted = tuple(
                row for row in related if row.lifecycle is EventLifecycle.ACCEPTED
            )
            committed = tuple(
                row
                for row in related
                if row.lifecycle is EventLifecycle.TRANSACTION_COMPLETED
                and row.processing_sequence == sequence
            )
            if (
                len(accepted) != 1
                or len(committed) != 1
                or len(
                    [
                        row
                        for row in related
                        if row.lifecycle is EventLifecycle.COMPLETED
                    ]
                )
                != 1
                or any(row.lifecycle is EventLifecycle.FAILED for row in related)
                or any(
                    row.event_type is not AgentEventType.CHAT
                    or row.source is not AgentEventSource.API_CHAT
                    for row in (*accepted, terminal, *committed)
                )
            ):
                raise QualifiedInteractionUnavailable("R07 Chat is not terminal")

            experience_id = experience_id_for_event(eid, sequence)
            stored = self._experience.load_committed_current(experience_id)
            record = stored.record
            if (
                record.lifecycle is not ExperienceLifecycle.ACTIVE
                or record.experience_id != experience_id
                or record.source_event_id != eid
                or record.source_event_sequence != sequence
            ):
                raise QualifiedInteractionUnavailable("R12 Experience invalid")
            db1 = self._memory.get_committed_episodic(record.source_episode_id)
            if db1 is None:
                raise QualifiedInteractionUnavailable("DB1 episode unavailable")
            episode = db1.record
            if (
                episode.record_type is not MemoryRecordType.EPISODIC_LOG
                or episode.archived
                or episode.context_id != record.context_id
                or episode.source_channel != "chat"
                or episode.source_session_id is None
                or len(episode.user_input.encode("utf-8")) > _MAX_MESSAGE_BYTES
                or len(episode.response.encode("utf-8")) > _MAX_MESSAGE_BYTES
            ):
                raise QualifiedInteractionUnavailable("DB1/Experience mismatch")

            # One code-owned coordination criterion, not generated "I trust".
            user = _Q1_USER.fullmatch(episode.user_input)
            self_answer = _Q1_SELF.fullmatch(episode.response)
            if (
                user is None
                or self_answer is None
                or user.group(1) == self_answer.group(1)
            ):
                raise QualifiedInteractionUnavailable(
                    "Q1 has no unresolved joint option selection"
                )

            wal = self._wal.inspect()
            manifest = wal.active_manifest
            if (
                not wal.exists
                or manifest is None
                or manifest.external_reconciliation_required
            ):
                raise QualifiedInteractionUnavailable("active WAL source unavailable")
            matches = [
                (index, row)
                for index, row in enumerate(wal.records)
                if isinstance(row, TransitionRecord)
                and str(row.event_id) == eid
                and row.processing_sequence == sequence
                and row.event_type == AgentEventType.CHAT.value
                and row.event_source == AgentEventSource.API_CHAT.value
            ]
            if len(matches) != 1:
                raise QualifiedInteractionUnavailable(
                    "historical event not retained in WAL"
                )
            index, transition = matches[0]
            if index < 1 or transition.candidate_snapshot.schema_version != 10:
                raise QualifiedInteractionUnavailable("event-time WAL unavailable")
            previous = wal.records[index - 1]
            if not isinstance(previous, (BaselineRecord, TransitionRecord)):
                raise QualifiedInteractionUnavailable("prior WAL record unavailable")
            prior = (
                previous.candidate_snapshot
                if isinstance(previous, TransitionRecord)
                else previous.baseline_snapshot
            )
            candidate = transition.candidate_snapshot
            if (
                transition.previous_record_hash != previous.record_hash
                or transition.prior_snapshot_sequence
                != prior.last_processed_event_sequence
                or candidate.last_processed_event_sequence != sequence
                or candidate.context_state.current_context_id != record.context_id
            ):
                raise QualifiedInteractionUnavailable("WAL event lineage mismatch")
            frames = tuple(
                frame
                for frame in candidate.context_state.frames
                if frame.context_id == record.context_id
            )
            previous_frames = tuple(
                frame
                for frame in prior.context_state.frames
                if frame.context_id == record.context_id
            )
            if len(frames) != 1 or len(previous_frames) > 1:
                raise QualifiedInteractionUnavailable("event-time Context ambiguous")
            frame = frames[0]
            past_tokens = previous_frames[0].participant_refs if previous_frames else ()
            if (
                frame.status != ContextStatus.ACTIVE.value
                or frame.source_channel != episode.source_channel
                or frame.source_session_id != episode.source_session_id
                or len(frame.participant_refs) != 1
                or past_tokens
            ):
                raise QualifiedInteractionUnavailable("not a first-bound Context token")

            # Any later event may change Context identity. A later ordinary
            # or debug Chat in the same Context is ineligible even if the
            # Context frame stays byte-identical.
            for later in wal.records[index + 1 :]:
                if not isinstance(later, TransitionRecord):
                    raise QualifiedInteractionUnavailable("WAL history rebased")
                newer_state = later.candidate_snapshot.context_state
                newer_frames = tuple(
                    item
                    for item in newer_state.frames
                    if item.context_id == record.context_id
                )
                if newer_frames != (frame,):
                    raise QualifiedInteractionUnavailable("Context modified after Q1")
                if (
                    later.event_type
                    in (
                        AgentEventType.CHAT.value,
                        AgentEventType.DEBUG_CHAT.value,
                    )
                    and newer_state.current_context_id == record.context_id
                ):
                    raise QualifiedInteractionUnavailable("later Chat supersedes Q1")
            current_context = self._contexts.state
            frame_now = next(
                (
                    current
                    for current in current_context.frames
                    if current.context_id == record.context_id
                ),
                None,
            )
            # WAL stores ContextFrameSnapshot, live R09 returns ContextFrame.
            # Compare via the existing validated registry projection rather
            # than cross-class dataclass equality (always False).
            projected = candidate.context_state.to_registry_state()
            expected = next(
                (
                    current
                    for current in projected.frames
                    if current.context_id == record.context_id
                ),
                None,
            )
            if (
                expected is None
                or frame_now != expected
                or current_context.current_context_id != record.context_id
            ):
                raise QualifiedInteractionUnavailable("current Context differs from E")
            observed = EventRef(eid, sequence, accepted[0].timestamp)
            if (
                observed.occurred_at > event.requested_at
                or observed.event_sequence >= event.processing_sequence
            ):
                raise QualifiedInteractionUnavailable("source event later than review")

            digest = experience_record_digest(record)
            root_token = sha256(
                canonical_json(
                    {
                        "source_event_id": eid,
                        "source_event_sequence": sequence,
                        "experience_digest": digest,
                        "operation_digest": stored.operation_digest,
                        "terminal_record_hash": terminal.record_hash,
                    }
                )
            ).hexdigest()
            witness = SourceWitness(
                SourceKind.EXPERIENCE,
                experience_id,
                record.revision,
                digest,
                root_token,
                SourceLifecycle.ACTIVE,
                ClaimedOrigin.SUBJECT_OBSERVATION,
                observed,
            )
            after = self._store.load()
            if (
                not isinstance(after, AgentStateSnapshotV10)
                or after != before
                or self._port._capture_for_agent_state().relationship.canonical_value()
                != after.relationship_state
            ):
                raise QualifiedInteractionUnavailable("committed U2 root changed")
            return QualifiedQ1Coordination(
                EventRef(
                    event.event_id,
                    event.processing_sequence,
                    event.requested_at,
                ),
                observed,
                witness,
                frame.participant_refs[0],
                record.context_id,
                user.group(1),
                self_answer.group(1),
            )
        except (
            AgentStateLoadError,
            EventJournalError,
            ExperienceStoreError,
            EpisodicMemoryReadError,
            EpisodicMemoryFormatError,
            StateWALError,
            QualifiedInteractionUnavailable,
            TypeError,
            AttributeError,
            ValueError,
            OSError,
            RecursionError,
        ) as exc:
            raise QualifiedInteractionUnavailable(
                "source-qualified Q1 interaction unavailable"
            ) from exc
