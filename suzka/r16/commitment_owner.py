"""Private R16 U2c source-owned Commitment proposal and independent acceptance.

One trustworthy cooperative SUZKA process may accept a concrete, finite
responsibility: produce a source-checked review artifact for a committed
Experience within seven days of a separate committed request. This is an
obligation, not a claim of completion, scheduler installation or Tool authority.
R07/R12/DB1 remain source owners, R13 remains Commitment mutation authority.
No caller/model proof, transcript or public API becomes intrinsic responsibility.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from suzka.experience import experience_record_digest
from suzka.identifiers import validate_identifier
from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.experience_store import ExperienceStoreError
from suzka.motivation.commitment import (
    CommitmentAdmissionReason,
    CommitmentLifecycle,
    CommitmentRecord,
    CommitmentRevisionOperation,
    CommitmentRevisionReason,
    CommitmentRevisionRecord,
    CommitmentSubjectAdmission,
    commitment_id_for_fields,
)
from suzka.motivation.commitment_system import (
    CommitmentMutationEvidence,
    CommitmentSystemEventOperation,
    CommitmentSystemSnapshot,
)
from suzka.motivation.common import Deadline, R13Reference, R13ReferenceKind
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
from suzka.r16.outcomes import (
    R16OwnerOutcome,
    R16OwnerReason,
    R16OwnerStatus,
    owner_reason,
)
from suzka.runtime.journal_source import (
    JournalSourceReason,
    JournalSourceStage,
    SourceUnavailable,
    archived_high_water,
    reason_of as _reason_of,
    locate_journal_source,
)

if TYPE_CHECKING:
    from suzka.runtime.main_loop import SuzkaMainLoop


class _CommitmentSourceUnavailable(SourceUnavailable):
    """Private, source-redacted refusal with no caller-provided evidence."""


# This exact bounded format is read only from the committed request's DB1 body.
_REQUEST_PREFIX = "COMMIT_REVIEW_EXPERIENCE:1:"
_SUBJECT = "Produce a source-checked review record"
_BENEFICIARY = R13Reference(R13ReferenceKind.SUBJECT, "subject:self")
_SCOPE = ("record_review_explanation", "record_source_qualification")
_OBLIGATION_WINDOW = timedelta(days=7)


def _utcnow() -> datetime:
    """Trusted local runtime clock; never derived from a request or model."""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _CommitmentSourceSelection:
    """Only source selectors, never a proposed Commitment or admission proof."""

    proposal: _MotivationSourceRef
    target: _MotivationSourceRef

    def __post_init__(self) -> None:
        if (
            type(self.proposal) is not _MotivationSourceRef
            or type(self.target) is not _MotivationSourceRef
        ):
            raise _CommitmentSourceUnavailable("invalid source selection")
        self.proposal.__post_init__()
        self.target.__post_init__()


@dataclass(frozen=True, slots=True)
class _QualifiedCommitment:
    proposal_event_id: str
    target_experience_id: str
    motivation_id: str
    motivation_event_id: str
    motivation_revision_digest: str
    motivation_record_digest: str
    proposal_record_digest: str
    target_record_digest: str
    context_id: str
    request_time: datetime
    due_time: datetime

    @property
    def evidence(self) -> tuple[R13Reference, ...]:
        values = (
            R13Reference(R13ReferenceKind.USER_REQUEST, self.proposal_event_id),
            R13Reference(R13ReferenceKind.EXPERIENCE, self.target_experience_id),
            R13Reference(R13ReferenceKind.MOTIVATION, self.motivation_id),
            R13Reference(R13ReferenceKind.EVENT, self.motivation_event_id),
        )
        if len({item.reference for item in values}) != len(values):
            raise _CommitmentSourceUnavailable("ambiguous source reference identity")
        return tuple(sorted(values, key=lambda item: (item.reference, item.kind.value)))

    @property
    def deadline(self) -> Deadline:
        return Deadline(self.due_time)

    @property
    def origins(self) -> tuple[R13Reference, ...]:
        return (R13Reference(R13ReferenceKind.USER_REQUEST, self.proposal_event_id),)


@dataclass(frozen=True, slots=True)
class _PendingCommitment:
    event: AgentEvent
    action: str
    selection: _CommitmentSourceSelection | None
    qualification: _QualifiedCommitment | None
    commitment_id: str | None
    snapshot: CommitmentSystemSnapshot


class _CommitmentSourceOwner:
    """Private app-bound, nonce-bound proposer and separate responsibility reviewer."""

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
            raise _CommitmentSourceUnavailable("Commitment source owners unavailable")
        self._loop = loop
        self._runtime = runtime
        self._journal = journal
        self._source = source_owner
        self._inflight: tuple[object, str, object] | None = None
        self._pending: _PendingCommitment | None = None
        loop._bind_commitment_source_owner(self)

    def _check_binding(self) -> None:
        if (
            self._loop._runtime is not self._runtime
            or self._loop._commitment_source_owner is not self
            or self._loop._motivation_source_owner is not self._source
        ):
            raise _CommitmentSourceUnavailable("Commitment source binding changed")
        self._source._check_binding()

    def _submit_proposal(
        self, selection: _CommitmentSourceSelection
    ) -> Future[AgentEventOutcome[R16OwnerOutcome]]:
        if type(selection) is not _CommitmentSourceSelection:
            raise _CommitmentSourceUnavailable("not a closed source selector")
        selection.__post_init__()
        return self._submit("proposal", selection)

    def _submit_review(self, commitment_id: str) -> Future[AgentEventOutcome[R16OwnerOutcome]]:
        validate_identifier(commitment_id)
        return self._submit("review", commitment_id)

    def _submit(self, action: str, subject: object) -> Future[AgentEventOutcome[R16OwnerOutcome]]:
        self._check_binding()
        if action not in ("proposal", "review"):
            raise _CommitmentSourceUnavailable("invalid private operation")
        nonce = object()
        kind, origin = self._event_kind(action)

        def handle() -> R16OwnerOutcome:
            if self._pending is not None or self._inflight is not None:
                raise _CommitmentSourceUnavailable("Commitment event already pending")
            self._inflight = (nonce, action, subject)
            try:
                if action == "proposal":
                    return self._loop._propose_commitment_source(self, nonce)
                return self._loop._review_commitment_source(self, nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(kind, origin, handle)

    @staticmethod
    def _event_kind(action: str) -> tuple[AgentEventType, AgentEventSource]:
        if action == "proposal":
            return (
                AgentEventType.COMMITMENT_PROPOSAL_REVIEW,
                AgentEventSource.RUNTIME_COMMITMENT_PROPOSAL_REVIEW,
            )
        return (
            AgentEventType.COMMITMENT_ACCEPTANCE_REVIEW,
            AgentEventSource.RUNTIME_COMMITMENT_ACCEPTANCE_REVIEW,
        )

    def _event(self, loop: SuzkaMainLoop, nonce: object, action: str) -> AgentEvent:
        self._check_binding()
        event = self._runtime.current_event()
        expected_type, expected_source = self._event_kind(action)
        if (
            loop is not self._loop
            or self._inflight is None
            or self._inflight[0] is not nonce
            or self._inflight[1] != action
            or self._pending is not None
            or event is None
            or event.event_type is not expected_type
            or event.source is not expected_source
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _CommitmentSourceUnavailable("unbound Commitment worker")
        return event

    def _qualify(
        self,
        selection: _CommitmentSourceSelection,
        event: AgentEvent,
        *,
        continuation: bool = False,
    ) -> _QualifiedCommitment:
        """Require actual earlier X, independently created U2a M, and later P.

        A proposal admits new sources, which must be retained; acceptance
        review and Plan continue an admitted Commitment, so archived sources
        are accepted on committed owner evidence (IP-4c).
        """
        self._check_binding()
        p, x = selection.proposal, selection.target
        if p.event_id == x.event_id or p.event_sequence <= x.event_sequence:
            raise _CommitmentSourceUnavailable("invalid source chronology")
        try:
            x_record, x_source = self._source._read(
                x, event, continuation=continuation
            )
            p_record, p_source = self._source._read(
                p, event, continuation=continuation
            )
            p_db1 = self._source._memory.get_committed_episodic(p.episode_id)
            if (
                p_db1 is None
                or p_db1.record.user_input
                != f"{_REQUEST_PREFIX}{x_record.experience_id}"
                or p_source.context_id != x_source.context_id
                or p_source.source_channel != x_source.source_channel
                or p_source.source_session_id != x_source.source_session_id
                or p_record.context_id != x_record.context_id
            ):
                raise _CommitmentSourceUnavailable("unqualified request or Context")
            due = p_record.created_at + _OBLIGATION_WINDOW
            # R07 requested_at is a historical queue-admission timestamp, not
            # execution time. A queued review can cross the obligation deadline;
            # check the trusted current UTC clock *both* in the handler and in
            # this same qualifier at pre-WAL validation. Do not rewrite R07 time.
            if event.requested_at >= due or _utcnow() >= due:
                raise _CommitmentSourceUnavailable(
                    "obligation deadline elapsed",
                    reason=R16OwnerReason.DEADLINE_ELAPSED,
                )

            target = R13Reference(R13ReferenceKind.EXPERIENCE, x_record.experience_id)
            snap = self._loop._motivation_system.snapshot()
            interest = tuple(
                m
                for m in snap.records
                if m.kind is MotivationKind.INTEREST and m.target == target
            )
            if len(interest) != 1:
                raise _CommitmentSourceUnavailable(
                    "independent INTEREST absent",
                    reason=R16OwnerReason.MOTIVATION_UNAVAILABLE,
                )
            motivation = interest[0]
            if (
                motivation.lifecycle is not MotivationLifecycle.ACTIVE
                or motivation.source_evidence != target
                or motivation.conflict_refs
                or motivation.revision != 0
                or len(motivation.revision_history) != 1
            ):
                raise _CommitmentSourceUnavailable(
                    "Motivation changed or conflicted",
                    reason=R16OwnerReason.MOTIVATION_UNAVAILABLE,
                )
            if any(
                other.kind is MotivationKind.AVERSION
                and other.target == target
                and other.lifecycle is MotivationLifecycle.ACTIVE
                for other in snap.records
            ):
                raise _CommitmentSourceUnavailable(
                    "counter-Motivation active",
                    reason=R16OwnerReason.COUNTER_MOTIVATION,
                )
            revision = motivation.revision_history[0]
            entries = tuple(
                e
                for e in snap.evidence_ledger
                if e.motivation_id == motivation.motivation_id
                and e.evidence.source_ref == target
            )
            if len(entries) != 1:
                raise _CommitmentSourceUnavailable("Motivation ledger unavailable")
            entry = entries[0]
            receipts = tuple(
                r
                for r in snap.event_receipts
                if r.event_id == entry.event_id
                and r.event_sequence == entry.event_sequence
            )
            if len(receipts) != 1:
                raise _CommitmentSourceUnavailable("Motivation receipt unavailable")
            receipt = receipts[0]
            if (
                receipt.operation is not MotivationEventOperation.EVIDENCE
                or receipt.motivation_ids != (motivation.motivation_id,)
                or receipt.evidence_refs != (target,)
                or receipt.evidence_digest != entry.evidence.evidence_digest
                or entry.evidence.kind is not MotivationKind.INTEREST
                or entry.evidence.target != target
                or entry.evidence.authority_witness_digest
                != experience_record_digest(x_record)
                or entry.evidence.source_event_id != x.event_id
                or entry.evidence.source_event_sequence != x.event_sequence
                or entry.evidence.observed_at != x_record.created_at
                or entry.evidence.origin_refs
                != (R13Reference(R13ReferenceKind.EVENT, x.event_id),)
                or revision.event_id != entry.event_id
                or revision.event_sequence != entry.event_sequence
                or not (x.event_sequence < entry.event_sequence < p.event_sequence)
            ):
                raise _CommitmentSourceUnavailable("pre-P M witness invalid")

            inspection = self._journal.inspect()
            if (
                inspection.external_reconciliation_required
                or inspection.open_transactions
                or inspection.reconciliation_required_transactions
            ):
                raise _CommitmentSourceUnavailable("Journal unresolved")
            stage, u2a = locate_journal_source(
                inspection, entry.event_id, entry.event_sequence
            )
            if stage is JournalSourceStage.MISSING or (
                stage is JournalSourceStage.ARCHIVED and not continuation
            ):
                raise _CommitmentSourceUnavailable(
                    "U2a event not retained",
                    reason=JournalSourceReason.SOURCE_MISSING
                    if stage is JournalSourceStage.MISSING
                    else JournalSourceReason.SOURCE_PAST_RETENTION,
                )
            # An archived U2a event is backed by the R13 receipt checked above.
            for lifecycle in (
                ()
                if stage is JournalSourceStage.ARCHIVED
                else (EventLifecycle.ACCEPTED, EventLifecycle.COMPLETED)
            ):
                rows = tuple(row for row in u2a if row.lifecycle is lifecycle)
                if (
                    len(rows) != 1
                    or rows[0].event_type is not AgentEventType.MOTIVATION_SOURCE_REVIEW
                    or rows[0].source
                    is not AgentEventSource.RUNTIME_MOTIVATION_SOURCE_REVIEW
                    or (
                        lifecycle is EventLifecycle.COMPLETED
                        and rows[0].processing_sequence != entry.event_sequence
                    )
                ):
                    raise _CommitmentSourceUnavailable("U2a event not terminal")
            if any(row.lifecycle is EventLifecycle.FAILED for row in u2a):
                raise _CommitmentSourceUnavailable("U2a event failed")
            result = _QualifiedCommitment(
                proposal_event_id=p.event_id,
                target_experience_id=x_record.experience_id,
                motivation_id=motivation.motivation_id,
                motivation_event_id=entry.event_id,
                motivation_revision_digest=revision.record_digest,
                motivation_record_digest=motivation.record_digest,
                proposal_record_digest=experience_record_digest(p_record),
                target_record_digest=experience_record_digest(x_record),
                context_id=x_record.context_id,
                request_time=p_record.created_at,
                due_time=due,
            )
            _ = result.evidence  # Reject same identifier in distinct evidence kinds.
            return result
        except _CommitmentSourceUnavailable:
            raise
        except (
            ValueError,
            RuntimeError,
            OSError,
            OverflowError,
            ExperienceStoreError,
            EventJournalError,
        ) as error:
            raise _CommitmentSourceUnavailable(
                "Commitment source unavailable", reason=_reason_of(error)
            ) from error

    @staticmethod
    def _assert_contract(record: CommitmentRecord, q: _QualifiedCommitment) -> None:
        if (
            record.subject != _SUBJECT
            or record.beneficiary != _BENEFICIARY
            or record.scope != _SCOPE
            or record.deadline != q.deadline
            or record.origin_refs != q.origins
            or record.evidence_refs != q.evidence
            or record.related_goal_refs
            or record.desire_refs
        ):
            raise _CommitmentSourceUnavailable("Commitment terms altered")

    def _selection_from_record(
        self, record: CommitmentRecord
    ) -> _CommitmentSourceSelection:
        if (
            record.subject != _SUBJECT
            or record.beneficiary != _BENEFICIARY
            or record.scope != _SCOPE
            or record.related_goal_refs
            or record.desire_refs
            or len(record.origin_refs) != 1
            or record.origin_refs[0].kind is not R13ReferenceKind.USER_REQUEST
        ):
            raise _CommitmentSourceUnavailable("unrecognized Commitment contract")
        p_id = record.origin_refs[0].reference
        x_refs = tuple(
            ref
            for ref in record.evidence_refs
            if ref.kind is R13ReferenceKind.EXPERIENCE
        )
        if len(x_refs) != 1:
            raise _CommitmentSourceUnavailable("Commitment target unavailable")
        try:
            inspection = self._journal.inspect()
            x_record = self._source._experience.load_committed_current(
                x_refs[0].reference
            ).record
            relevant = tuple(row for row in inspection.records if row.event_id == p_id)
            if relevant:
                completed = tuple(
                    row
                    for row in relevant
                    if row.lifecycle is EventLifecycle.COMPLETED
                )
                if len(completed) != 1 or completed[0].processing_sequence is None:
                    raise _CommitmentSourceUnavailable("request not terminal")
                p_seq = completed[0].processing_sequence
            else:
                # IP-4c continuation: the archived request lies after the
                # target and inside the archived range, before the genesis.
                genesis = record.revision_history[0]
                high_water = archived_high_water(inspection)
                if genesis.event_sequence is None or high_water < 1:
                    raise _CommitmentSourceUnavailable(
                        "request not terminal",
                        reason=JournalSourceReason.SOURCE_MISSING,
                    )
                p_seq = self._source._archived_source_sequence(
                    p_id,
                    x_record.source_event_sequence,
                    min(genesis.event_sequence, high_water + 1),
                )
            p_record = self._source._experience.load_committed_current(
                experience_id_for_event(p_id, p_seq)
            ).record
            return _CommitmentSourceSelection(
                _MotivationSourceRef(p_id, p_seq, p_record.source_episode_id),
                _MotivationSourceRef(
                    x_record.source_event_id,
                    x_record.source_event_sequence,
                    x_record.source_episode_id,
                ),
            )
        except _CommitmentSourceUnavailable:
            raise
        except (
            ValueError,
            OSError,
            RuntimeError,
            ExperienceStoreError,
            EventJournalError,
        ) as error:
            raise _CommitmentSourceUnavailable(
                "Committed proposal source missing", reason=_reason_of(error)
            ) from error

    def _assert_original_proposal(self, record: CommitmentRecord) -> None:
        genesis = record.revision_history[0]
        if genesis.event_id is None or genesis.event_sequence is None:
            raise _CommitmentSourceUnavailable("Commitment proposal event missing")
        ingestions = tuple(
            receipt
            for receipt in self._loop._commitment_system.snapshot().event_receipts
            if receipt.operation is CommitmentSystemEventOperation.INGEST_PROPOSAL
            and receipt.commitment_id == record.commitment_id
        )
        if len(ingestions) != 1:
            raise _CommitmentSourceUnavailable("Commitment proposal receipt missing")
        receipt = ingestions[0]
        if (
            receipt.event_id != genesis.event_id
            or receipt.event_sequence != genesis.event_sequence
            or receipt.proposal_digest != record.proposal_digest
            or receipt.proposal_genesis != genesis
        ):
            raise _CommitmentSourceUnavailable("Commitment proposal receipt mismatched")
        stage, relevant = locate_journal_source(
            self._journal.inspect(), genesis.event_id, genesis.event_sequence
        )
        if stage is JournalSourceStage.MISSING:
            raise _CommitmentSourceUnavailable(
                "Commitment proposal event missing",
                reason=JournalSourceReason.SOURCE_MISSING,
            )
        # An archived proposal event is backed by the bound receipt above.
        for lifecycle in (
            ()
            if stage is JournalSourceStage.ARCHIVED
            else (EventLifecycle.ACCEPTED, EventLifecycle.COMPLETED)
        ):
            rows = tuple(row for row in relevant if row.lifecycle is lifecycle)
            if (
                len(rows) != 1
                or rows[0].event_type is not AgentEventType.COMMITMENT_PROPOSAL_REVIEW
                or rows[0].source
                is not AgentEventSource.RUNTIME_COMMITMENT_PROPOSAL_REVIEW
                or (
                    lifecycle is EventLifecycle.COMPLETED
                    and rows[0].processing_sequence != genesis.event_sequence
                )
            ):
                raise _CommitmentSourceUnavailable("Commitment proposal not terminal")
        if any(row.lifecycle is EventLifecycle.FAILED for row in relevant):
            raise _CommitmentSourceUnavailable("Commitment proposal event failed")

    def _other_responsibility_exists(
        self, q: _QualifiedCommitment, *, excluding: str | None = None
    ) -> bool:
        exact_x = R13Reference(R13ReferenceKind.EXPERIENCE, q.target_experience_id)
        return any(
            record.commitment_id != excluding
            and record.beneficiary == _BENEFICIARY
            and record.subject == _SUBJECT
            and record.scope == _SCOPE
            and exact_x in record.evidence_refs
            for record in self._loop._commitment_system.records
        )

    def _propose(self, loop: SuzkaMainLoop, nonce: object) -> R16OwnerOutcome:
        event = self._event(loop, nonce, "proposal")
        assert self._inflight is not None
        selection = self._inflight[2]
        if type(selection) is not _CommitmentSourceSelection:
            raise _CommitmentSourceUnavailable("invalid Commitment source selection")
        before = loop._commitment_system.snapshot()
        try:
            q = self._qualify(selection, event)
            if self._other_responsibility_exists(q):
                raise _CommitmentSourceUnavailable(
                    "duplicate responsibility",
                    reason=R16OwnerReason.COMMITMENT_ALREADY_EXISTS,
                )
        except _CommitmentSourceUnavailable as error:
            # IP-5: an unqualified proposal is a HOLD with a closed reason,
            # not a FAILED event. Nothing was mutated.
            self._pending = _PendingCommitment(event, "hold", None, None, None, before)
            return R16OwnerOutcome.hold(owner_reason(error))
        evidence = q.evidence
        commitment_id = commitment_id_for_fields(
            subject=_SUBJECT,
            beneficiary=_BENEFICIARY,
            scope=_SCOPE,
            deadline=q.deadline,
            evidence_refs=evidence,
            origin_refs=q.origins,
        )
        assert event.processing_sequence is not None
        genesis = CommitmentRevisionRecord(
            commitment_id=commitment_id,
            revision=0,
            operation=CommitmentRevisionOperation.CREATE,
            reason=CommitmentRevisionReason.CREATION,
            created_at=event.requested_at,
            previous_lifecycle_state=None,
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            evidence_refs=tuple(sorted(item.reference for item in evidence)),
        )
        proposal = CommitmentRecord(
            subject=_SUBJECT,
            beneficiary=_BENEFICIARY,
            scope=_SCOPE,
            deadline=q.deadline,
            evidence_refs=evidence,
            origin_refs=q.origins,
            revision_history=(genesis,),
        )
        loop._commitment_system.ingest_proposal(
            proposal,
            CommitmentMutationEvidence(
                event.event_id, event.processing_sequence, event.requested_at, evidence
            ),
        )
        self._pending = _PendingCommitment(
            event,
            "proposal",
            selection,
            q,
            proposal.commitment_id,
            loop._commitment_system.snapshot(),
        )
        return R16OwnerOutcome(R16OwnerStatus.PROPOSED, proposal.commitment_id)

    def _review(self, loop: SuzkaMainLoop, nonce: object) -> R16OwnerOutcome:
        event = self._event(loop, nonce, "review")
        assert self._inflight is not None
        identifier = self._inflight[2]
        if type(identifier) is not str:
            raise _CommitmentSourceUnavailable("invalid responsibility reference")
        before = loop._commitment_system.snapshot()
        commitment = loop._commitment_system.get(identifier)
        if (
            commitment is None
            or commitment.lifecycle is not CommitmentLifecycle.PROPOSED
        ):
            self._pending = _PendingCommitment(event, "hold", None, None, None, before)
            return R16OwnerOutcome.hold(R16OwnerReason.NOT_PENDING_REVIEW)
        try:
            selection = self._selection_from_record(commitment)
            q = self._qualify(selection, event, continuation=True)
            self._assert_contract(commitment, q)
            self._assert_original_proposal(commitment)
            if self._other_responsibility_exists(q, excluding=commitment.commitment_id):
                raise _CommitmentSourceUnavailable(
                    "responsibility already exists",
                    reason=R16OwnerReason.COMMITMENT_ALREADY_EXISTS,
                )
        except _CommitmentSourceUnavailable as error:
            self._pending = _PendingCommitment(
                event, "hold", None, None, identifier, before
            )
            return R16OwnerOutcome.hold(owner_reason(error), identifier)
        assert event.processing_sequence is not None
        admission = CommitmentSubjectAdmission(
            commitment_id=commitment.commitment_id,
            proposal_digest=commitment.proposal_digest,
            beneficiary=commitment.beneficiary,
            scope=commitment.scope,
            deadline=commitment.deadline,
            evidence_refs=tuple(
                sorted(ref.reference for ref in commitment.evidence_refs)
            ),
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            reason=CommitmentAdmissionReason.SUBJECT_RESPONSIBILITY,
        )
        loop._commitment_system.accept(
            identifier,
            admission,
            CommitmentMutationEvidence(
                event.event_id,
                event.processing_sequence,
                event.requested_at,
                commitment.evidence_refs,
            ),
        )
        self._pending = _PendingCommitment(
            event,
            "accept",
            selection,
            q,
            identifier,
            loop._commitment_system.snapshot(),
        )
        return R16OwnerOutcome(R16OwnerStatus.ACCEPTED, identifier)

    def _validate_commit(self, event: AgentEvent) -> None:
        pending = self._pending
        if event.event_type not in {
            AgentEventType.COMMITMENT_PROPOSAL_REVIEW,
            AgentEventType.COMMITMENT_ACCEPTANCE_REVIEW,
        }:
            if pending is not None:
                raise _CommitmentSourceUnavailable("dangling responsibility event")
            return
        if pending is None or pending.event != event:
            raise _CommitmentSourceUnavailable("unqualified Commitment mutation")
        if self._loop._commitment_system.snapshot() != pending.snapshot:
            raise _CommitmentSourceUnavailable("Commitment state changed pre-WAL")
        if pending.action == "hold":
            return
        if pending.selection is None or pending.qualification is None:
            raise _CommitmentSourceUnavailable("qualified responsibility absent")
        current = (
            self._qualify(pending.selection, event, continuation=True)
            if pending.action == "accept"
            else self._qualify(pending.selection, event)
        )
        if current != pending.qualification:
            raise _CommitmentSourceUnavailable("Commitment source changed pre-WAL")
        commitment = self._loop._commitment_system.get(pending.commitment_id or "")
        if commitment is None:
            raise _CommitmentSourceUnavailable("Commitment absent pre-WAL")
        self._assert_contract(commitment, current)
        if pending.action == "proposal":
            if commitment.lifecycle is not CommitmentLifecycle.PROPOSED:
                raise _CommitmentSourceUnavailable("proposal illegally accepted")
        elif pending.action == "accept":
            self._assert_original_proposal(commitment)
            if self._other_responsibility_exists(
                current, excluding=commitment.commitment_id
            ):
                raise _CommitmentSourceUnavailable(
                    "duplicate responsibility before WAL"
                )
            if commitment.lifecycle is not CommitmentLifecycle.ACTIVE:
                raise _CommitmentSourceUnavailable("Commitment not accepted")

    def _clear(self, event: AgentEvent) -> None:
        del event
        self._inflight = None
        self._pending = None
