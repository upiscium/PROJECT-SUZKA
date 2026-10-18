"""R15 P1b-A: narrowly source-qualified, process-local referent proposals.

This module owns no state mutation. Only independently terminal R07/R12/DB1
records in one anonymous R09 Context may underwrite a bounded interpretation
of the subject's *own previous response* and a *later user-reported* referent.

Not a general semantic learner, verified human-intent oracle, R15 StateWAL
producer, historical SourceEpoch attestor, or an operator/model setter.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any

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
from suzka.runtime.context import (
    ContextNotFound,
    ContextRegistry,
    ContextStateInvalid,
    ContextStatus,
)
from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalError,
    EventLifecycle,
)

_MAX_MESSAGE_BYTES = 4096
_OPTION_DOMAIN = b"SUZKA:R15:P1BA:SOURCE-REFERENT:V1\x00"
_SETUP = re.compile(
    r"比較用の案は3つです。案Aは「[^」]{1,128}」、"
    r"案Bは「[^」]{1,128}」、新案Cは「[^」]{1,128}」です。"
    r"分類はまだしないでください。"
)
_PRIOR = (
    re.compile(r"暫定的に[^\n。]{0,30}案([AB])"),
    re.compile(r"案([AB])を暫定的に(?:選|選択)"),
    re.compile(r"前の案[（(]案([AB])[）)]"),
)
_LATER = re.compile(
    r"先の「前の案」について補足します。"
    r"今回の比較実験で事前に指定されていた参照先は案([AB])です。"
)


class QualifiedProposalUnavailable(ValueError):
    """A coarse refusal, without raw source/model/private text."""


@dataclass(frozen=True, slots=True)
class QualifiedTurnRef:
    """Identifiers only; never a source credential or caller-authored witness."""

    event_id: str
    event_sequence: int
    episode_id: str


@dataclass(frozen=True, slots=True)
class _ObservedTurn:
    event: EventRef
    episode_id: str
    context_id: str
    request: str
    response: str
    witness: SourceWitness


@dataclass(frozen=True, slots=True)
class QualifiedReferentProposal:
    """Derived inside the source gate from already committed owned records."""

    stage: str
    source_scope: str
    selected: str
    original_choice: str
    source: SourceWitness
    contrary: SourceWitness | None
    original_event: EventRef
    last_observed_event: EventRef
    context_id: str


def scoped_option_key(setup_episode_id: str, option: str) -> str:
    """Resolvable only with the original *qualified* finite setup scope."""
    if type(setup_episode_id) is not str or not setup_episode_id:
        raise QualifiedProposalUnavailable("source scope unavailable")
    if type(option) is not str or option not in ("A", "B", "UNKNOWN"):
        raise QualifiedProposalUnavailable("candidate option unavailable")
    return sha256(
        _OPTION_DOMAIN
        + canonical_json({"setup_episode_id": setup_episode_id, "option": option})
    ).hexdigest()


def restore_scoped_option(setup_episode_id: str, statement_id: str) -> str:
    options = (
        choice
        for choice in ("A", "B", "UNKNOWN")
        if scoped_option_key(setup_episode_id, choice) == statement_id
    )
    found = tuple(options)
    if len(found) != 1:
        raise QualifiedProposalUnavailable("source-scoped meaning unresolved")
    return found[0]


def _parse_candidate(value: str, phase: str) -> str:
    if type(value) is not str or len(value.encode("utf-8")) > _MAX_MESSAGE_BYTES:
        raise QualifiedProposalUnavailable("candidate unavailable")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, val in pairs:
            if key in result:
                raise QualifiedProposalUnavailable("duplicate candidate field")
            result[key] = val
        return result

    try:
        data = json.loads(
            value,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, TypeError, RecursionError) as error:
        raise QualifiedProposalUnavailable("invalid candidate record") from error
    if type(data) is not dict or set(data) != {
        "schema",
        "phase",
        "assumed_referent",
        "assessment",
        "uncertainty",
        "evidence_steps",
        "rationale",
    }:
        raise QualifiedProposalUnavailable("candidate shape unavailable")
    expected_steps = ["E"] if phase == "S1" else ["E", "S1", "E2"]
    if (
        type(data["schema"]) is not int
        or data["schema"] != 1
        or data["phase"] != phase
        or type(data["phase"]) is not str
        or type(data["assumed_referent"]) is not str
        or data["assumed_referent"] not in ("A", "B", "UNKNOWN")
        or type(data["evidence_steps"]) is not list
        or data["evidence_steps"] != expected_steps
        or type(data["assessment"]) is not str
        or data["assessment"] not in ("LIKELY_ALIGNED", "MAY_HAVE_MISREAD", "UNKNOWN")
        or type(data["uncertainty"]) is not str
        or data["uncertainty"] not in ("LOW", "MEDIUM", "HIGH")
        or type(data["rationale"]) is not str
        or not 0 < len(data["rationale"].encode("utf-8")) <= 700
    ):
        raise QualifiedProposalUnavailable("candidate source or phase invalid")
    return data["assumed_referent"]


class QualifiedNarrativeSourceGate:
    """One cooperative, runtime-bound local source reader: NO public setter.

    Accepts only owner objects, reopens and qualifies every event at each
    admission, and makes no assertions about participant identity.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        journal: EventJournal,
        experience: ExperienceStore,
        memory: DualMemorySystem,
        contexts: ContextRegistry,
    ) -> None:
        if (
            type(runtime) is not AgentRuntime
            or type(journal) is not EventJournal
            or type(experience) is not ExperienceStore
            or type(memory) is not DualMemorySystem
            or type(contexts) is not ContextRegistry
            or not journal.has_exclusive_authority
        ):
            raise QualifiedProposalUnavailable("bound source ownership unavailable")
        if (
            experience.root.resolve()
            != ExperienceStore.from_memory_root(
                memory.settings.memory.persist_directory
            ).root.resolve()
        ):
            raise QualifiedProposalUnavailable(
                "source owners do not share a store root"
            )
        self._runtime = runtime
        self._journal = journal
        self._experience = experience
        self._memory = memory
        self._contexts = contexts

    def current_review_event(self) -> EventRef:
        ev: AgentEvent | None = self._runtime.current_event()
        if (
            ev is None
            or ev.event_type is not AgentEventType.NARRATIVE_REVIEW
            or ev.source is not AgentEventSource.RUNTIME_NARRATIVE_REVIEW
            or type(ev.processing_sequence) is not int
            or ev.processing_sequence < 1
        ):
            raise QualifiedProposalUnavailable(
                "not an internal serialized narrative review"
            )
        return EventRef(ev.event_id, ev.processing_sequence, ev.requested_at)

    def _read(self, source: QualifiedTurnRef) -> _ObservedTurn:
        if type(source) is not QualifiedTurnRef:
            raise QualifiedProposalUnavailable("untrusted source identity shape")
        if type(source.event_id) is not str or type(source.episode_id) is not str:
            raise QualifiedProposalUnavailable("untrusted source identity type")
        if type(source.event_sequence) is not int or source.event_sequence < 1:
            raise QualifiedProposalUnavailable("untrusted source order")
        # Source owners re-validate this ID; a valid-looking Ref is NOT authority.
        try:
            inspected = self._journal.inspect()
            if (
                inspected.external_reconciliation_required
                or inspected.open_transactions
                or inspected.reconciliation_required_transactions
            ):
                raise QualifiedProposalUnavailable("Journal source is in recovery")
            relevant = tuple(
                r for r in inspected.records if r.event_id == source.event_id
            )
            accepted = tuple(
                r for r in relevant if r.lifecycle is EventLifecycle.ACCEPTED
            )
            terminal = tuple(
                r
                for r in relevant
                if r.lifecycle is EventLifecycle.COMPLETED
                and r.processing_sequence == source.event_sequence
            )
            transaction = tuple(
                r
                for r in relevant
                if r.lifecycle is EventLifecycle.TRANSACTION_COMPLETED
                and r.processing_sequence == source.event_sequence
            )
            if (
                len(accepted) != 1
                or len(terminal) != 1
                or len(transaction) != 1
                or any(r.lifecycle is EventLifecycle.FAILED for r in relevant)
                or any(
                    r.event_type is not AgentEventType.CHAT
                    or r.source is not AgentEventSource.API_CHAT
                    for r in (*accepted, *terminal, *transaction)
                )
            ):
                raise QualifiedProposalUnavailable("R07 event not terminal")
            experience_id = experience_id_for_event(
                source.event_id, source.event_sequence
            )
            entry = self._experience.load_committed_current(experience_id)
            record = entry.record
            if (
                record.lifecycle is not ExperienceLifecycle.ACTIVE
                or record.source_event_id != source.event_id
                or record.source_event_sequence != source.event_sequence
                or record.source_episode_id != source.episode_id
                or record.experience_id != experience_id
            ):
                raise QualifiedProposalUnavailable("R12 Experience is inconsistent")
            committed = self._memory.get_committed_episodic(source.episode_id)
            if committed is None:
                raise QualifiedProposalUnavailable("DB1 episode not committed")
            episode = committed.record
            if (
                episode.record_type is not MemoryRecordType.EPISODIC_LOG
                or episode.archived
                or not episode.context_id
                or episode.context_id != record.context_id
                or len(episode.user_input.encode("utf-8")) > _MAX_MESSAGE_BYTES
                or len(episode.response.encode("utf-8")) > _MAX_MESSAGE_BYTES
            ):
                raise QualifiedProposalUnavailable("DB1/Experience mismatched")
            context = self._contexts.get(record.context_id)
            if (
                context.status is not ContextStatus.ACTIVE
                or episode.source_channel != context.source_channel
                or episode.source_session_id != context.source_session_id
                or context.participant_refs
            ):
                # Until event-time participant identity is actually captured,
                # anonymous context-only interpretation is all we can qualify.
                raise QualifiedProposalUnavailable("event-time context ambiguous")
            event = EventRef(
                record.source_event_id,
                record.source_event_sequence,
                accepted[0].timestamp,
            )
            root_token = sha256(
                canonical_json(
                    {
                        "source_event_id": record.source_event_id,
                        "source_event_sequence": record.source_event_sequence,
                        "experience_digest": experience_record_digest(record),
                        "operation_digest": entry.operation_digest,
                        "terminal_record_hash": terminal[0].record_hash,
                    }
                )
            ).hexdigest()
            witness = SourceWitness(
                SourceKind.EXPERIENCE,
                experience_id,
                record.revision,
                experience_record_digest(record),
                root_token,
                SourceLifecycle.ACTIVE,
                ClaimedOrigin.SUBJECT_OBSERVATION,
                event,
            )
            return _ObservedTurn(
                event,
                source.episode_id,
                record.context_id,
                episode.user_input,
                episode.response,
                witness,
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
            if isinstance(error, QualifiedProposalUnavailable):
                raise
            raise QualifiedProposalUnavailable(
                "source owner read unavailable"
            ) from error

    def qualify(
        self,
        turns: tuple[QualifiedTurnRef, ...],
        *,
        stage: str,
    ) -> QualifiedReferentProposal:
        reviewer = self.current_review_event()
        if stage not in ("S1", "S2") or len(turns) != (3 if stage == "S1" else 5):
            raise QualifiedProposalUnavailable("unexpected source stage")
        observed = tuple(self._read(turn) for turn in turns)
        sequences = tuple(item.event.event_sequence for item in observed)
        if (
            tuple(sorted(set(sequences))) != sequences
            or reviewer.event_sequence <= sequences[-1]
            or reviewer.occurred_at < observed[-1].event.occurred_at
            or len({item.context_id for item in observed}) != 1
        ):
            raise QualifiedProposalUnavailable("event order/context inconsistent")
        setup, first, s1 = observed[:3]
        if not _SETUP.fullmatch(setup.request):
            raise QualifiedProposalUnavailable(
                "original bounded option set unavailable"
            )
        # The initial response must demonstrably refer to one chosen option.
        original_matches = {
            selected
            for pattern in _PRIOR
            for selected in pattern.findall(first.response.replace("*", ""))
        }
        if len(original_matches) != 1:
            raise QualifiedProposalUnavailable("original answer has no single referent")
        original = original_matches.pop()
        pre_candidate = _parse_candidate(s1.response, "S1")
        if pre_candidate != original:
            raise QualifiedProposalUnavailable(
                "provisional S1 contradicts observed self-answer"
            )
        if stage == "S1":
            return QualifiedReferentProposal(
                "S1",
                setup.episode_id,
                original,
                original,
                first.witness,
                None,
                first.event,
                s1.event,
                first.context_id,
            )
        later, s2 = observed[3:]
        target = _LATER.fullmatch(later.request)
        next_candidate = _parse_candidate(s2.response, "S2")
        if (
            target is None
            or next_candidate != target.group(1)
            or next_candidate == original
        ):
            raise QualifiedProposalUnavailable("no supported new reinterpretation")
        return QualifiedReferentProposal(
            "S2",
            setup.episode_id,
            next_candidate,
            original,
            later.witness,
            first.witness,
            first.event,
            s2.event,
            first.context_id,
        )
