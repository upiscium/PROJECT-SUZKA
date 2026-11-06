"""Bounded R16 U2b private Goal proposal and independent subject review.

This is a trusted single-process/cooperative-writer MVI, not historical
Context-frame attestation or adversarial in-process issuer authentication.
External CHAT proposes a target; no caller or model can issue Goal admission.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING

from suzka.experience import experience_record_digest
from suzka.identifiers import validate_identifier
from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.experience_store import ExperienceStoreError
from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.goal import (
    GoalAdmissionReason,
    GoalLifecycle,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    GoalSubjectAdmission,
    goal_id_for_target,
    goal_proposal_digest,
)
from suzka.motivation.goal_system import (
    GoalMutationEvidence,
    GoalSystemEventOperation,
    GoalSystemSnapshot,
)
from suzka.motivation.motivation import MotivationKind, MotivationLifecycle
from suzka.motivation.system import MotivationEventOperation
from suzka.r16.source_owner import _MotivationSourceOwner, _MotivationSourceRef
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.event_journal import EventJournal, EventJournalError, EventLifecycle

if TYPE_CHECKING:
    from suzka.runtime.main_loop import SuzkaMainLoop


class _GoalSourceUnavailable(ValueError):
    """Bounded private HOLD cause, never source text."""


_DESCRIPTION = "Review committed experience"


@dataclass(frozen=True, slots=True)
class _GoalSourceSelection:
    """Only identifiers for two already-committed CHATs; never a Goal proof."""

    proposal: _MotivationSourceRef
    target: _MotivationSourceRef

    def __post_init__(self) -> None:
        if (
            type(self.proposal) is not _MotivationSourceRef
            or type(self.target) is not _MotivationSourceRef
        ):
            raise _GoalSourceUnavailable("invalid source selector")


@dataclass(frozen=True, slots=True)
class _QualifiedGoal:
    proposal_event_id: str
    target_experience_id: str
    motivation_id: str
    motivation_event_id: str
    motivation_revision_digest: str
    motivation_record_digest: str
    proposal_record_digest: str
    target_record_digest: str
    context_id: str

    @property
    def target(self) -> R13Reference:
        return R13Reference(R13ReferenceKind.EXPERIENCE, self.target_experience_id)

    @property
    def evidence(self) -> tuple[R13Reference, ...]:
        values = (
            R13Reference(R13ReferenceKind.USER_REQUEST, self.proposal_event_id),
            self.target,
            R13Reference(R13ReferenceKind.MOTIVATION, self.motivation_id),
            R13Reference(R13ReferenceKind.EVENT, self.motivation_event_id),
        )
        if len({x.reference for x in values}) != len(values):
            raise _GoalSourceUnavailable("source identity collision")
        return tuple(sorted(values, key=lambda x: (x.reference, x.kind.value)))


@dataclass(frozen=True, slots=True)
class _PendingGoal:
    event: AgentEvent
    action: str
    selection: _GoalSourceSelection | None
    qualification: _QualifiedGoal | None
    goal_id: str | None
    goal_snapshot: GoalSystemSnapshot


class _GoalSourceOwner:
    """The only app-bound presenter of source-selected R13 proposal/admission."""

    def __init__(
        self,
        loop: SuzkaMainLoop,
        runtime: AgentRuntime,
        journal: EventJournal,
        source_owner: _MotivationSourceOwner,
    ) -> None:
        from suzka.runtime.main_loop import SuzkaMainLoop

        if (
            not isinstance(loop, SuzkaMainLoop)
            or type(runtime) is not AgentRuntime
            or type(journal) is not EventJournal
            or type(source_owner) is not _MotivationSourceOwner
            or not journal.has_exclusive_authority
            or loop._runtime is not runtime
            or loop._motivation_source_owner is not source_owner
            or source_owner._journal is not journal
            or source_owner._runtime is not runtime
            or source_owner._loop is not loop
        ):
            raise _GoalSourceUnavailable("unqualified Goal Source owners")
        self._loop = loop
        self._runtime = runtime
        self._journal = journal
        self._source = source_owner
        self._inflight: tuple[object, str, object] | None = None
        self._pending: _PendingGoal | None = None
        loop._bind_goal_source_owner(self)

    def _check_binding(self) -> None:
        if (
            self._loop._runtime is not self._runtime
            or self._loop._goal_source_owner is not self
            or self._loop._motivation_source_owner is not self._source
        ):
            raise _GoalSourceUnavailable("Goal Source owner changed")
        self._source._check_binding()

    def _submit_proposal(
        self, selection: _GoalSourceSelection
    ) -> Future[AgentEventOutcome[str]]:
        if type(selection) is not _GoalSourceSelection:
            raise _GoalSourceUnavailable("source selector must be exact")
        selection.__post_init__()
        return self._submit("proposal", selection)

    def _submit_review(self, goal_id: str) -> Future[AgentEventOutcome[str]]:
        validate_identifier(goal_id)
        return self._submit("review", goal_id)

    def _submit(self, action: str, selection: object) -> Future[AgentEventOutcome[str]]:
        self._check_binding()
        nonce = object()
        event_type, event_source = (
            (
                AgentEventType.GOAL_PROPOSAL_REVIEW,
                AgentEventSource.RUNTIME_GOAL_PROPOSAL_REVIEW,
            )
            if action == "proposal"
            else (
                AgentEventType.GOAL_ADOPTION_REVIEW,
                AgentEventSource.RUNTIME_GOAL_ADOPTION_REVIEW,
            )
        )

        def handle() -> str:
            if self._inflight is not None or self._pending is not None:
                raise _GoalSourceUnavailable("Goal source already pending")
            self._inflight = (nonce, action, selection)
            try:
                if action == "proposal":
                    return self._loop._propose_goal_source(self, nonce)
                return self._loop._review_goal_source(self, nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(event_type, event_source, handle)

    def _current_event(
        self, loop: SuzkaMainLoop, nonce: object, action: str
    ) -> AgentEvent:
        self._check_binding()
        event = self._runtime.current_event()
        expected_type, expected_source = (
            (
                AgentEventType.GOAL_PROPOSAL_REVIEW,
                AgentEventSource.RUNTIME_GOAL_PROPOSAL_REVIEW,
            )
            if action == "proposal"
            else (
                AgentEventType.GOAL_ADOPTION_REVIEW,
                AgentEventSource.RUNTIME_GOAL_ADOPTION_REVIEW,
            )
        )
        if (
            loop is not self._loop
            or self._inflight is None
            or self._inflight[0] is not nonce
            or self._inflight[1] != action
            or self._pending is not None
            or event is None
            or event.event_type is not expected_type
            or event.source is not expected_source
            or event.processing_sequence is None
        ):
            raise _GoalSourceUnavailable("unbound Goal review event")
        return event

    def _qualify(
        self, selection: _GoalSourceSelection, event: AgentEvent
    ) -> _QualifiedGoal:
        """Re-read retained real R07/R12/DB1 sources; no caller proof is ingested."""
        self._check_binding()
        p, x = selection.proposal, selection.target
        if p.event_id == x.event_id or p.event_sequence <= x.event_sequence:
            raise _GoalSourceUnavailable("Goal source chronology invalid")
        try:
            x_record, x_source = self._source._read(x, event)
            p_record, p_source = self._source._read(p, event)
            db1_p = self._source._memory.get_committed_episodic(p.episode_id)
            if (
                db1_p is None
                or db1_p.record.user_input
                != f"REVIEW_EXPERIENCE:{x_record.experience_id}"
                or p_source.context_id != x_source.context_id
                or p_source.source_channel != x_source.source_channel
                or p_source.source_session_id != x_source.source_session_id
                or p_record.context_id != x_record.context_id
            ):
                raise _GoalSourceUnavailable("Goal proposal is not source-qualified")
            motivation = self._loop._motivation_system.snapshot()
            target = R13Reference(R13ReferenceKind.EXPERIENCE, x_record.experience_id)
            matches = tuple(
                m
                for m in motivation.records
                if m.kind is MotivationKind.INTEREST and m.target == target
            )
            if len(matches) != 1:
                raise _GoalSourceUnavailable("independent Motivation absent")
            m = matches[0]
            if (
                m.lifecycle is not MotivationLifecycle.ACTIVE
                or m.source_evidence != target
                or m.conflict_refs
                or len(m.revision_history) != 1
                or m.revision != 0
                or m.revision_history[-1].event_id is None
            ):
                raise _GoalSourceUnavailable("Motivation unqualified or changed")
            if any(
                other.kind is MotivationKind.AVERSION
                and other.target == target
                and other.lifecycle is MotivationLifecycle.ACTIVE
                for other in motivation.records
            ):
                raise _GoalSourceUnavailable("explicit counter-Motivation")
            revision = m.revision_history[-1]
            rows = tuple(
                entry
                for entry in motivation.evidence_ledger
                if entry.motivation_id == m.motivation_id
                and entry.evidence.source_ref == target
            )
            if len(rows) != 1:
                raise _GoalSourceUnavailable("Motivation source ledger ambiguous")
            entry = rows[0]
            receipts = tuple(
                receipt
                for receipt in motivation.event_receipts
                if receipt.event_id == entry.event_id
                and receipt.event_sequence == entry.event_sequence
            )
            if len(receipts) != 1:
                raise _GoalSourceUnavailable("Motivation receipt missing")
            receipt = receipts[0]
            if (
                receipt.operation is not MotivationEventOperation.EVIDENCE
                or receipt.evidence_digest != entry.evidence.evidence_digest
                or receipt.motivation_ids != (m.motivation_id,)
                or receipt.evidence_refs != (target,)
                or entry.evidence.kind is not MotivationKind.INTEREST
                or entry.evidence.target != target
                or entry.evidence.source_event_id != x.event_id
                or entry.evidence.source_event_sequence != x.event_sequence
                or entry.evidence.observed_at != x_record.created_at
                or entry.evidence.origin_refs
                != (R13Reference(R13ReferenceKind.EVENT, x.event_id),)
                or entry.evidence.authority_witness_digest
                != experience_record_digest(x_record)
                or entry.evidence.source_ref != target
                or revision.event_id != entry.event_id
                or revision.event_sequence != entry.event_sequence
                or not (x.event_sequence < entry.event_sequence < p.event_sequence)
            ):
                raise _GoalSourceUnavailable(
                    "Motivation historical revision unqualified"
                )
            inspected = self._journal.inspect()
            if (
                inspected.external_reconciliation_required
                or inspected.open_transactions
                or inspected.reconciliation_required_transactions
            ):
                raise _GoalSourceUnavailable("Journal recovery underway")
            provenance = tuple(
                row for row in inspected.records if row.event_id == entry.event_id
            )
            # U2a is an internal R13-only mutation, not a coordinated DB1
            # transaction: require its accepted/terminal event and R13 receipt,
            # not a fabricated external TRANSACTION_COMPLETED lifecycle.
            for lifecycle in (
                EventLifecycle.ACCEPTED,
                EventLifecycle.COMPLETED,
            ):
                matches_lifecycle = tuple(
                    row for row in provenance if row.lifecycle is lifecycle
                )
                if (
                    len(matches_lifecycle) != 1
                    or matches_lifecycle[0].event_type
                    is not AgentEventType.MOTIVATION_SOURCE_REVIEW
                    or matches_lifecycle[0].source
                    is not AgentEventSource.RUNTIME_MOTIVATION_SOURCE_REVIEW
                    or (
                        lifecycle is not EventLifecycle.ACCEPTED
                        and matches_lifecycle[0].processing_sequence
                        != entry.event_sequence
                    )
                ):
                    raise _GoalSourceUnavailable("Motivation owner event not terminal")
            if any(row.lifecycle is EventLifecycle.FAILED for row in provenance):
                raise _GoalSourceUnavailable("Motivation event failed")
            result = _QualifiedGoal(
                proposal_event_id=p.event_id,
                target_experience_id=x_record.experience_id,
                motivation_id=m.motivation_id,
                motivation_event_id=entry.event_id,
                motivation_revision_digest=revision.record_digest,
                motivation_record_digest=m.record_digest,
                proposal_record_digest=experience_record_digest(p_record),
                target_record_digest=experience_record_digest(x_record),
                context_id=x_record.context_id,
            )
            _ = result.evidence  # enforce cross-kind ID uniqueness before any write
            return result
        except _GoalSourceUnavailable:
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            raise _GoalSourceUnavailable("Goal source read unavailable") from exc

    def _selection_from_goal(self, goal: GoalRecord) -> _GoalSourceSelection:
        """Derive identifiers solely from immutable R13 proposal and R07/R12."""
        if (
            goal.description != _DESCRIPTION
            or goal.target.kind is not R13ReferenceKind.EXPERIENCE
            or goal.deadline.is_set
            or goal.dependencies
            or goal.conflicts
            or len(goal.origin_refs) != 1
            or goal.origin_refs[0].kind is not R13ReferenceKind.USER_REQUEST
        ):
            raise _GoalSourceUnavailable("unrecognized Goal proposal contract")
        p_id = goal.origin_refs[0].reference
        x_id = goal.target.reference
        try:
            inspection = self._journal.inspect()
            completed = tuple(
                row
                for row in inspection.records
                if row.event_id == p_id and row.lifecycle is EventLifecycle.COMPLETED
            )
            if len(completed) != 1 or completed[0].processing_sequence is None:
                raise _GoalSourceUnavailable("request event not retained")
            p_seq = completed[0].processing_sequence
            p_entry = self._source._experience.load_committed_current(
                experience_id_for_event(p_id, p_seq)
            ).record
            x_entry = self._source._experience.load_committed_current(x_id).record
            return _GoalSourceSelection(
                _MotivationSourceRef(p_id, p_seq, p_entry.source_episode_id),
                _MotivationSourceRef(
                    x_entry.source_event_id,
                    x_entry.source_event_sequence,
                    x_entry.source_episode_id,
                ),
            )
        except _GoalSourceUnavailable:
            raise
        except (
            ValueError,
            OSError,
            RuntimeError,
            ExperienceStoreError,
            EventJournalError,
        ) as exc:
            raise _GoalSourceUnavailable("Goal proposal sources unavailable") from exc

    @staticmethod
    def _expect_goal_contract(goal: GoalRecord, q: _QualifiedGoal) -> None:
        if (
            goal.description != _DESCRIPTION
            or goal.target != q.target
            or goal.origin_refs
            != (R13Reference(R13ReferenceKind.USER_REQUEST, q.proposal_event_id),)
            or goal.evidence_refs != q.evidence
            or goal.deadline.is_set
            or goal.dependencies
            or goal.conflicts
        ):
            raise _GoalSourceUnavailable("Goal proposal evidence changed")

    def _require_real_proposal_review(self, goal: GoalRecord) -> None:
        """Require the earlier PROPOSED Goal to originate in a terminal R16 event.

        This checks cooperative app provenance/receipts, not hostile in-process
        code impersonation. The proposal is a distinct earlier event, never the
        currently executing adoption review.
        """
        genesis = goal.revision_history[0]
        if genesis.event_id is None or genesis.event_sequence is None:
            raise _GoalSourceUnavailable("Goal genesis event missing")
        receipts = tuple(
            receipt
            for receipt in self._loop._goal_system.snapshot().event_receipts
            if receipt.operation is GoalSystemEventOperation.INGEST_PROPOSAL
            and receipt.goal_id == goal.goal_id
        )
        if len(receipts) != 1:
            raise _GoalSourceUnavailable("Goal proposal receipt missing")
        receipt = receipts[0]
        if (
            receipt.event_id != genesis.event_id
            or receipt.event_sequence != genesis.event_sequence
            or receipt.proposal_digest != goal.proposal_digest
            or receipt.proposal_genesis != genesis
        ):
            raise _GoalSourceUnavailable("Goal proposal receipt not bound")
        records = tuple(
            row
            for row in self._journal.inspect().records
            if row.event_id == genesis.event_id
        )
        for lifecycle in (EventLifecycle.ACCEPTED, EventLifecycle.COMPLETED):
            events = tuple(row for row in records if row.lifecycle is lifecycle)
            if (
                len(events) != 1
                or events[0].event_type is not AgentEventType.GOAL_PROPOSAL_REVIEW
                or events[0].source is not AgentEventSource.RUNTIME_GOAL_PROPOSAL_REVIEW
                or (
                    lifecycle is EventLifecycle.COMPLETED
                    and events[0].processing_sequence != genesis.event_sequence
                )
            ):
                raise _GoalSourceUnavailable("Goal proposal event not terminal")
        if any(row.lifecycle is EventLifecycle.FAILED for row in records):
            raise _GoalSourceUnavailable("Goal proposal event failed")

    def _propose(self, loop: SuzkaMainLoop, nonce: object) -> str:
        event = self._current_event(loop, nonce, "proposal")
        assert self._inflight is not None
        selection = self._inflight[2]
        if type(selection) is not _GoalSourceSelection:
            raise _GoalSourceUnavailable("invalid Goal source selector")
        q = self._qualify(selection, event)
        target = q.target
        goal_id = goal_id_for_target(target)
        if loop._goal_system.get(goal_id) is not None:
            raise _GoalSourceUnavailable("Goal already proposed for target")
        evidence = q.evidence
        assert event.processing_sequence is not None
        proposal_digest = goal_proposal_digest(
            target,
            _DESCRIPTION,
            origin_refs=(
                R13Reference(R13ReferenceKind.USER_REQUEST, q.proposal_event_id),
            ),
            evidence_refs=evidence,
        )
        genesis = GoalRevisionRecord(
            goal_id=goal_id,
            revision=0,
            operation=GoalRevisionOperation.CREATE,
            reason=GoalRevisionReason.CREATION,
            created_at=event.requested_at,
            previous_lifecycle_state=None,
            proposal_digest=proposal_digest,
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            evidence_refs=tuple(sorted(r.reference for r in evidence)),
        )
        goal = GoalRecord(
            goal_id=goal_id,
            target=target,
            description=_DESCRIPTION,
            origin_refs=(
                R13Reference(R13ReferenceKind.USER_REQUEST, q.proposal_event_id),
            ),
            evidence_refs=evidence,
            revision_history=(genesis,),
        )
        loop._goal_system.ingest_proposal(
            goal,
            GoalMutationEvidence(
                event.event_id, event.processing_sequence, event.requested_at, evidence
            ),
        )
        self._pending = _PendingGoal(
            event, "proposal", selection, q, goal_id, loop._goal_system.snapshot()
        )
        return goal_id

    def _review(self, loop: SuzkaMainLoop, nonce: object) -> str:
        event = self._current_event(loop, nonce, "review")
        assert self._inflight is not None
        goal_id = self._inflight[2]
        if type(goal_id) is not str:
            raise _GoalSourceUnavailable("invalid Goal review selection")
        goal = loop._goal_system.get(goal_id)
        before = loop._goal_system.snapshot()
        if goal is None or goal.lifecycle is not GoalLifecycle.PROPOSED:
            self._pending = _PendingGoal(event, "hold", None, None, None, before)
            return "hold"
        try:
            selection = self._selection_from_goal(goal)
            q = self._qualify(selection, event)
            self._expect_goal_contract(goal, q)
            self._require_real_proposal_review(goal)
            # Only an explicit conflict linked to this Goal/Motivation may
            # veto adoption. Global Value pairs (e.g. care vs honesty) have
            # no source-qualified link to Experience X in this first MVI;
            # inventing one would create a false negative or false authority.
            if any(
                other.lifecycle is GoalLifecycle.ADOPTED and goal_id in other.conflicts
                for other in self._loop._goal_system.records
                if other.goal_id != goal_id
            ):
                raise _GoalSourceUnavailable("an adopted Goal conflicts")
        except _GoalSourceUnavailable:
            self._pending = _PendingGoal(event, "hold", None, None, goal_id, before)
            return "hold"
        assert event.processing_sequence is not None
        proof = GoalSubjectAdmission(
            goal_id=goal.goal_id,
            proposal_digest=goal.proposal_digest,
            evidence_refs=tuple(sorted(x.reference for x in goal.evidence_refs)),
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            reason=GoalAdmissionReason.SUBJECT_REVIEW,
        )
        loop._goal_system.adopt(
            goal_id,
            proof,
            GoalMutationEvidence(
                event.event_id,
                event.processing_sequence,
                event.requested_at,
                goal.evidence_refs,
            ),
        )
        self._pending = _PendingGoal(
            event, "adopt", selection, q, goal_id, loop._goal_system.snapshot()
        )
        return "adopted"

    def _validate_commit(self, event: AgentEvent) -> None:
        pending = self._pending
        if event.event_type not in {
            AgentEventType.GOAL_PROPOSAL_REVIEW,
            AgentEventType.GOAL_ADOPTION_REVIEW,
        }:
            if pending is not None:
                raise _GoalSourceUnavailable("stale Goal Source pending")
            return
        if pending is None or pending.event != event:
            raise _GoalSourceUnavailable("unqualified Goal operation")
        if self._loop._goal_system.snapshot() != pending.goal_snapshot:
            raise _GoalSourceUnavailable("Goal state changed before WAL")
        if pending.action == "hold":
            return
        if pending.selection is None or pending.qualification is None:
            raise _GoalSourceUnavailable("Goal qualification missing")
        fresh = self._qualify(pending.selection, event)
        if fresh != pending.qualification:
            raise _GoalSourceUnavailable("Goal source changed before WAL")
        goal = self._loop._goal_system.get(pending.goal_id or "")
        if goal is None:
            raise _GoalSourceUnavailable("Goal disappeared before WAL")
        self._expect_goal_contract(goal, fresh)
        if pending.action == "adopt":
            self._require_real_proposal_review(goal)
        if (
            pending.action == "proposal"
            and goal.lifecycle is not GoalLifecycle.PROPOSED
        ):
            raise _GoalSourceUnavailable("unexpected Goal proposal mutation")
        if pending.action == "adopt":
            if goal.lifecycle is not GoalLifecycle.ADOPTED:
                raise _GoalSourceUnavailable("Goal review not admitted")

    def _clear(self, event: AgentEvent) -> None:
        del event
        self._inflight = None
        self._pending = None
