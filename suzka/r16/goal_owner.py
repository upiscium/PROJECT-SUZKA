"""Bounded R16 U2b private Goal proposal and independent subject review.

This is a trusted single-process/cooperative-writer MVI, not historical
Context-frame attestation or adversarial in-process issuer authentication.
External CHAT proposes a target; no caller or model can issue Goal admission.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING

from suzka.experience import ExperienceRecord, experience_record_digest
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
from suzka.motivation.motivation import (
    MotivationKind,
    MotivationLifecycle,
    MotivationRecord,
    MotivationRevisionRecord,
)
from suzka.motivation.system import (
    MotivationEventOperation,
    MotivationEvidenceLedgerEntry,
)
from suzka.r16.activation import self_goal_activation
from suzka.r16.review_value import ReviewValueUnavailable, qualify_review_value
from suzka.r16.source_owner import (
    _MotivationSourceOwner,
    _MotivationSourceRef,
    _MotivationSourceUnavailable,
)
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


class _GoalSourceUnavailable(SourceUnavailable):
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
class _SelfGoalSelection:
    """IP-11: only the committed CHAT X whose INTEREST may ground a self Goal."""

    target: _MotivationSourceRef

    def __post_init__(self) -> None:
        if type(self.target) is not _MotivationSourceRef:
            raise _GoalSourceUnavailable("invalid self-Goal source selector")


class GoalOrigin(str, Enum):
    """#379: who originated the Goal (the Explanation/projection value)."""

    USER_REQUEST = "user_request"
    SUBJECT_MOTIVATION = "subject_motivation"


AnyGoalSelection = _GoalSourceSelection | _SelfGoalSelection


@dataclass(frozen=True, slots=True)
class _QualifiedGoal:
    # None for a self-origin Goal: no user request P exists (IP-11).
    proposal_event_id: str | None
    target_experience_id: str
    motivation_id: str
    motivation_event_id: str
    motivation_revision_digest: str
    motivation_record_digest: str
    proposal_record_digest: str | None
    target_record_digest: str
    context_id: str
    # IP-10 (#371 D2 P2-3): X's own event and revision, from this qualified
    # record (the issuer's R15 read and Experience ref use them).
    target_event_id: str
    target_event_sequence: int
    target_revision: int
    origin: GoalOrigin = GoalOrigin.USER_REQUEST

    def __post_init__(self) -> None:
        if (self.origin is GoalOrigin.USER_REQUEST) != (
            self.proposal_event_id is not None
        ):
            raise _GoalSourceUnavailable("Goal origin and request disagree")

    @property
    def target(self) -> R13Reference:
        return R13Reference(R13ReferenceKind.EXPERIENCE, self.target_experience_id)

    @property
    def origin_refs(self) -> tuple[R13Reference, ...]:
        if self.proposal_event_id is None:
            # #379 D1 P0-2: a self Goal originates in its own Motivation.
            return (R13Reference(R13ReferenceKind.MOTIVATION, self.motivation_id),)
        return (R13Reference(R13ReferenceKind.USER_REQUEST, self.proposal_event_id),)

    @property
    def evidence(self) -> tuple[R13Reference, ...]:
        values = (
            *(
                ()
                if self.proposal_event_id is None
                else (R13Reference(R13ReferenceKind.USER_REQUEST, self.proposal_event_id),)
            ),
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
    selection: AnyGoalSelection | None
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
    ) -> Future[AgentEventOutcome[R16OwnerOutcome]]:
        if type(selection) is not _GoalSourceSelection:
            raise _GoalSourceUnavailable("source selector must be exact")
        selection.__post_init__()
        return self._submit("proposal", selection)

    def _submit_review(self, goal_id: str) -> Future[AgentEventOutcome[R16OwnerOutcome]]:
        validate_identifier(goal_id)
        return self._submit("review", goal_id)

    def _submit_self_proposal(
        self, selection: _SelfGoalSelection
    ) -> Future[AgentEventOutcome[R16OwnerOutcome]]:
        """IP-11 (#379 D1 P1-4): a separate, later event after U2a."""

        if type(selection) is not _SelfGoalSelection:
            raise _GoalSourceUnavailable("self-Goal selector must be exact")
        selection.__post_init__()
        return self._submit("self_proposal", selection)

    @staticmethod
    def _event_kind(action: str) -> tuple[AgentEventType, AgentEventSource]:
        if action == "proposal":
            return (
                AgentEventType.GOAL_PROPOSAL_REVIEW,
                AgentEventSource.RUNTIME_GOAL_PROPOSAL_REVIEW,
            )
        if action == "self_proposal":
            return (
                AgentEventType.GOAL_SELF_PROPOSAL_REVIEW,
                AgentEventSource.RUNTIME_GOAL_SELF_PROPOSAL_REVIEW,
            )
        return (
            AgentEventType.GOAL_ADOPTION_REVIEW,
            AgentEventSource.RUNTIME_GOAL_ADOPTION_REVIEW,
        )

    def _submit(self, action: str, selection: object) -> Future[AgentEventOutcome[R16OwnerOutcome]]:
        self._check_binding()
        nonce = object()
        event_type, event_source = self._event_kind(action)

        def handle() -> R16OwnerOutcome:
            if self._inflight is not None or self._pending is not None:
                raise _GoalSourceUnavailable("Goal source already pending")
            self._inflight = (nonce, action, selection)
            try:
                if action == "proposal":
                    return self._loop._propose_goal_source(self, nonce)
                if action == "self_proposal":
                    return self._loop._propose_self_goal_source(self, nonce)
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
            or event.processing_sequence is None
        ):
            raise _GoalSourceUnavailable("unbound Goal review event")
        return event

    def _qualify(
        self,
        selection: _GoalSourceSelection,
        event: AgentEvent,
        *,
        continuation: bool = False,
    ) -> _QualifiedGoal:
        """Re-read real R07/R12/DB1 sources; no caller proof is ingested.

        A proposal admits new sources, so they must be retained. Adoption and
        Plan continue an admitted Goal: archived sources are accepted on the
        committed Experience and R13 receipts (IP-4c, #381 D2 P1-3).
        """
        self._check_binding()
        p, x = selection.proposal, selection.target
        if p.event_id == x.event_id or p.event_sequence <= x.event_sequence:
            raise _GoalSourceUnavailable("Goal source chronology invalid")
        try:
            # The request P is read first, so its outcome never depends on
            # the requester-named target X (IP-5 review P2-a).
            p_record, p_source = self._source._read(
                p, event, continuation=continuation
            )
            try:
                x_record, x_source = self._source._read(
                    x, event, continuation=continuation
                )
            except _MotivationSourceUnavailable as error:
                # A target-only reason is released only within P's Context.
                raise _GoalSourceUnavailable(
                    "target source unavailable",
                    reason=self._source._scoped_target_reason(
                        error, x, p, event, continuation=continuation
                    ),
                ) from error
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
            m, entry, revision = self._qualify_motivation(
                x, x_record, upper=p.event_sequence, continuation=continuation
            )
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
                target_event_id=x_record.source_event_id,
                target_event_sequence=x_record.source_event_sequence,
                target_revision=x_record.revision,
            )
            _ = result.evidence  # enforce cross-kind ID uniqueness before any write
            return result
        except _GoalSourceUnavailable:
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            raise _GoalSourceUnavailable(
                "Goal source read unavailable", reason=_reason_of(exc)
            ) from exc

    def _qualify_self(
        self,
        selection: _SelfGoalSelection,
        event: AgentEvent,
        *,
        continuation: bool = False,
        proposal: bool = False,
    ) -> _QualifiedGoal:
        """IP-11 (#379 D1-D3, D5): an INTEREST on X may ground a self Goal.

        Source qualification only.  The budget and "X has no Goal" are
        checked once in ``_propose_self`` (D5 P2-1), never here, so the
        WAL-before re-qualification cannot count its own proposal.
        """
        self._check_binding()
        x = selection.target
        settings = self._loop.settings.self_goal
        if event.processing_sequence is None:
            raise _GoalSourceUnavailable("self-Goal event not sequenced")
        try:
            inspected = self._journal.inspect()
            marker = self_goal_activation(inspected)
            if marker is None or marker.processing_sequence is None:
                raise _GoalSourceUnavailable(
                    "self-Goal path not activated",
                    reason=R16OwnerReason.SELF_GOAL_NOT_ACTIVATED,
                )
            if (
                inspected.external_reconciliation_required
                or inspected.open_transactions
                or inspected.reconciliation_required_transactions
            ):
                raise _GoalSourceUnavailable("Journal recovery underway")
            # X is a committed CHAT Experience only (#379 D1 P2): U2a never
            # reads anything else, so no self Goal can ground another.
            x_record, _x_source = self._source._read(
                x, event, continuation=continuation
            )
            m, entry, revision = self._qualify_motivation(
                x, x_record, upper=event.processing_sequence, continuation=continuation
            )
            # Only Motivations from U2a events after the marker (D2 P2-2).
            if entry.event_sequence <= marker.processing_sequence:
                raise _GoalSourceUnavailable(
                    "Motivation precedes self-Goal activation",
                    reason=R16OwnerReason.SELF_GOAL_NOT_ACTIVATED,
                )
            if m.strength < settings.min_strength:
                raise _GoalSourceUnavailable(
                    "INTEREST below self-Goal threshold",
                    reason=R16OwnerReason.MOTIVATION_BELOW_THRESHOLD,
                )
            if proposal:
                created = m.revision_history[0].created_at
                if event.requested_at.astimezone(timezone.utc) - created.astimezone(
                    timezone.utc
                ) > timedelta(hours=settings.stale_after_hours):
                    raise _GoalSourceUnavailable(
                        "INTEREST too old for a self Goal",
                        reason=R16OwnerReason.MOTIVATION_STALE,
                    )
            # A user's responsibility on X comes first (#379 D3 P1-1).
            target = R13Reference(R13ReferenceKind.EXPERIENCE, x_record.experience_id)
            if any(
                target in record.evidence_refs
                for record in self._loop._commitment_system.snapshot().records
            ):
                raise _GoalSourceUnavailable(
                    "a Commitment already names the target",
                    reason=R16OwnerReason.TARGET_HAS_COMMITMENT,
                )
            # The Plan's Value must be qualified now (D5 P2-3).
            qualify_review_value(self._loop, x_record.context_id)
            result = _QualifiedGoal(
                proposal_event_id=None,
                target_experience_id=x_record.experience_id,
                motivation_id=m.motivation_id,
                motivation_event_id=entry.event_id,
                motivation_revision_digest=revision.record_digest,
                motivation_record_digest=m.record_digest,
                proposal_record_digest=None,
                target_record_digest=experience_record_digest(x_record),
                context_id=x_record.context_id,
                target_event_id=x_record.source_event_id,
                target_event_sequence=x_record.source_event_sequence,
                target_revision=x_record.revision,
                origin=GoalOrigin.SUBJECT_MOTIVATION,
            )
            _ = result.evidence
            return result
        except _GoalSourceUnavailable:
            raise
        except _MotivationSourceUnavailable as error:
            raise _GoalSourceUnavailable(
                "self-Goal target unavailable", reason=owner_reason(error)
            ) from error
        except ReviewValueUnavailable as error:
            raise _GoalSourceUnavailable(
                "review Value not qualified", reason=R16OwnerReason.SOURCE_UNAVAILABLE
            ) from error
        except (ValueError, OSError, RuntimeError) as exc:
            raise _GoalSourceUnavailable(
                "self-Goal source read unavailable", reason=_reason_of(exc)
            ) from exc

    def _qualify_selection(
        self,
        selection: AnyGoalSelection,
        event: AgentEvent,
        *,
        continuation: bool = False,
    ) -> _QualifiedGoal:
        """Dispatch by origin (#379 D1 P1-1); Plan preview uses this too."""

        if type(selection) is _SelfGoalSelection:
            return self._qualify_self(selection, event, continuation=continuation)
        if type(selection) is _GoalSourceSelection:
            return self._qualify(selection, event, continuation=continuation)
        raise _GoalSourceUnavailable("unknown Goal source selector")

    def _qualify_motivation(
        self,
        x: _MotivationSourceRef,
        x_record: ExperienceRecord,
        *,
        upper: int,
        continuation: bool,
    ) -> tuple[MotivationRecord, MotivationEvidenceLedgerEntry, MotivationRevisionRecord]:
        """The one INTEREST on X and its U2a provenance (shared by U2b/U2b-S).

        ``upper`` bounds the U2a event: before the request P (user path) or
        before the current event (self path, never the same event).
        """
        motivation = self._loop._motivation_system.snapshot()
        target = R13Reference(R13ReferenceKind.EXPERIENCE, x_record.experience_id)
        matches = tuple(
            m
            for m in motivation.records
            if m.kind is MotivationKind.INTEREST and m.target == target
        )
        if len(matches) != 1:
            raise _GoalSourceUnavailable(
                "independent Motivation absent",
                reason=R16OwnerReason.MOTIVATION_UNAVAILABLE,
            )
        m = matches[0]
        if (
            m.lifecycle is not MotivationLifecycle.ACTIVE
            or m.source_evidence != target
            or m.conflict_refs
            or len(m.revision_history) != 1
            or m.revision != 0
            or m.revision_history[-1].event_id is None
        ):
            raise _GoalSourceUnavailable(
                "Motivation unqualified or changed",
                reason=R16OwnerReason.MOTIVATION_UNAVAILABLE,
            )
        if any(
            other.kind is MotivationKind.AVERSION
            and other.target == target
            and other.lifecycle is MotivationLifecycle.ACTIVE
            for other in motivation.records
        ):
            raise _GoalSourceUnavailable(
                "explicit counter-Motivation",
                reason=R16OwnerReason.COUNTER_MOTIVATION,
            )
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
            or not (x.event_sequence < entry.event_sequence < upper)
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
        stage, provenance = locate_journal_source(
            inspected, entry.event_id, entry.event_sequence
        )
        if stage is JournalSourceStage.MISSING or (
            stage is JournalSourceStage.ARCHIVED and not continuation
        ):
            raise _GoalSourceUnavailable(
                "Motivation owner event not retained",
                reason=JournalSourceReason.SOURCE_MISSING
                if stage is JournalSourceStage.MISSING
                else JournalSourceReason.SOURCE_PAST_RETENTION,
            )
        # U2a is an internal R13-only mutation, not a coordinated DB1
        # transaction: require its accepted/terminal event and R13 receipt,
        # not a fabricated external TRANSACTION_COMPLETED lifecycle. An
        # archived U2a event is backed by the R13 receipt checked above.
        for lifecycle in (
            ()
            if stage is JournalSourceStage.ARCHIVED
            else (EventLifecycle.ACCEPTED, EventLifecycle.COMPLETED)
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
        return m, entry, revision

    @staticmethod
    def _goal_origin(goal: GoalRecord) -> GoalOrigin:
        if (
            len(goal.origin_refs) == 1
            and goal.origin_refs[0].kind is R13ReferenceKind.MOTIVATION
        ):
            return GoalOrigin.SUBJECT_MOTIVATION
        return GoalOrigin.USER_REQUEST

    def _selection_from_goal(self, goal: GoalRecord) -> AnyGoalSelection:
        """Derive identifiers solely from immutable R13 proposal and R07/R12."""
        if self._goal_origin(goal) is GoalOrigin.SUBJECT_MOTIVATION:
            return self._self_selection_from_goal(goal)
        return self._user_selection_from_goal(goal)

    def _self_selection_from_goal(self, goal: GoalRecord) -> _SelfGoalSelection:
        if (
            goal.description != _DESCRIPTION
            or goal.target.kind is not R13ReferenceKind.EXPERIENCE
            or goal.deadline.is_set
            or goal.dependencies
            or goal.conflicts
        ):
            raise _GoalSourceUnavailable("unrecognized self-Goal contract")
        try:
            x_entry = self._source._experience.load_committed_current(
                goal.target.reference
            ).record
        except (ValueError, OSError, RuntimeError, ExperienceStoreError) as exc:
            raise _GoalSourceUnavailable(
                "self-Goal target unavailable", reason=_reason_of(exc)
            ) from exc
        return _SelfGoalSelection(
            _MotivationSourceRef(
                x_entry.source_event_id,
                x_entry.source_event_sequence,
                x_entry.source_episode_id,
            )
        )

    def _user_selection_from_goal(self, goal: GoalRecord) -> _GoalSourceSelection:
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
            x_entry = self._source._experience.load_committed_current(x_id).record
            relevant = tuple(row for row in inspection.records if row.event_id == p_id)
            if relevant:
                completed = tuple(
                    row
                    for row in relevant
                    if row.lifecycle is EventLifecycle.COMPLETED
                )
                if len(completed) != 1 or completed[0].processing_sequence is None:
                    raise _GoalSourceUnavailable("request event not retained")
                p_seq = completed[0].processing_sequence
            else:
                # IP-4c continuation: the request CHAT was archived. Its
                # sequence lies after the target's and inside the archived
                # range, before the Goal genesis.
                genesis = goal.revision_history[0]
                high_water = archived_high_water(inspection)
                if genesis.event_sequence is None or high_water < 1:
                    raise _GoalSourceUnavailable(
                        "request event missing",
                        reason=JournalSourceReason.SOURCE_MISSING,
                    )
                p_seq = self._source._archived_source_sequence(
                    p_id,
                    x_entry.source_event_sequence,
                    min(genesis.event_sequence, high_water + 1),
                )
            p_entry = self._source._experience.load_committed_current(
                experience_id_for_event(p_id, p_seq)
            ).record
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
            raise _GoalSourceUnavailable(
                "Goal proposal sources unavailable", reason=_reason_of(exc)
            ) from exc

    @staticmethod
    def _expect_goal_contract(goal: GoalRecord, q: _QualifiedGoal) -> None:
        if (
            goal.description != _DESCRIPTION
            or goal.target != q.target
            or goal.origin_refs != q.origin_refs
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
        stage, records = locate_journal_source(
            self._journal.inspect(), genesis.event_id, genesis.event_sequence
        )
        if stage is JournalSourceStage.MISSING:
            raise _GoalSourceUnavailable(
                "Goal proposal event missing",
                reason=JournalSourceReason.SOURCE_MISSING,
            )
        # An archived proposal event is backed by the bound receipt above.
        expected_type, expected_source = self._event_kind(
            "self_proposal"
            if self._goal_origin(goal) is GoalOrigin.SUBJECT_MOTIVATION
            else "proposal"
        )
        for lifecycle in (
            ()
            if stage is JournalSourceStage.ARCHIVED
            else (EventLifecycle.ACCEPTED, EventLifecycle.COMPLETED)
        ):
            events = tuple(row for row in records if row.lifecycle is lifecycle)
            if (
                len(events) != 1
                or events[0].event_type is not expected_type
                or events[0].source is not expected_source
                or (
                    lifecycle is EventLifecycle.COMPLETED
                    and events[0].processing_sequence != genesis.event_sequence
                )
            ):
                raise _GoalSourceUnavailable("Goal proposal event not terminal")
        if any(row.lifecycle is EventLifecycle.FAILED for row in records):
            raise _GoalSourceUnavailable("Goal proposal event failed")

    def _propose(self, loop: SuzkaMainLoop, nonce: object) -> R16OwnerOutcome:
        event = self._current_event(loop, nonce, "proposal")
        assert self._inflight is not None
        selection = self._inflight[2]
        if type(selection) is not _GoalSourceSelection:
            raise _GoalSourceUnavailable("invalid Goal source selector")
        before = loop._goal_system.snapshot()
        try:
            q = self._qualify(selection, event)
            self._require_target_free(loop, q.target)
        except _GoalSourceUnavailable as error:
            # IP-5: an unqualified proposal is a HOLD with a closed reason,
            # not a FAILED event. Nothing was mutated.
            self._pending = _PendingGoal(event, "hold", None, None, None, before)
            return R16OwnerOutcome.hold(owner_reason(error))
        return self._ingest(loop, event, "proposal", selection, q)

    def _propose_self(self, loop: SuzkaMainLoop, nonce: object) -> R16OwnerOutcome:
        """IP-11 (#379): a PROPOSED self-origin Goal, never its adoption."""

        event = self._current_event(loop, nonce, "self_proposal")
        assert self._inflight is not None
        selection = self._inflight[2]
        if type(selection) is not _SelfGoalSelection:
            raise _GoalSourceUnavailable("invalid self-Goal source selector")
        before = loop._goal_system.snapshot()
        try:
            q = self._qualify_self(selection, event, proposal=True)
            self._require_target_free(loop, q.target)
            # Once per proposal, outside qualification (D5 P2-1).
            self._require_daily_budget(loop, event)
        except _GoalSourceUnavailable as error:
            self._pending = _PendingGoal(event, "hold", None, None, None, before)
            return R16OwnerOutcome.hold(owner_reason(error))
        return self._ingest(loop, event, "self_proposal", selection, q)

    @staticmethod
    def _require_target_free(loop: SuzkaMainLoop, target: R13Reference) -> None:
        existing = loop._goal_system.get(goal_id_for_target(target))
        if existing is not None:
            raise _GoalSourceUnavailable(
                "Goal already proposed for target",
                # IP-5 semantics kept: only a Goal a user asked for is
                # GOAL_ALREADY_EXISTS; any other Goal is the subject's own.
                reason=R16OwnerReason.GOAL_ALREADY_EXISTS
                if any(
                    ref.kind is R13ReferenceKind.USER_REQUEST
                    for ref in existing.origin_refs
                )
                else R16OwnerReason.SELF_GOAL_OCCUPIES_TARGET,
            )

    def _require_daily_budget(self, loop: SuzkaMainLoop, event: AgentEvent) -> None:
        """#379 D2 P0: self Goals per UTC day, from their genesis ``created_at``."""

        day = event.requested_at.astimezone(timezone.utc).date()
        count = sum(
            1
            for goal in loop._goal_system.records
            if self._goal_origin(goal) is GoalOrigin.SUBJECT_MOTIVATION
            and goal.revision_history[0].created_at.astimezone(timezone.utc).date()
            == day
        )
        if count >= loop.settings.self_goal.per_day:
            raise _GoalSourceUnavailable(
                "self-Goal daily budget exhausted",
                reason=R16OwnerReason.SELF_GOAL_BUDGET_EXHAUSTED,
            )

    def _ingest(
        self,
        loop: SuzkaMainLoop,
        event: AgentEvent,
        action: str,
        selection: AnyGoalSelection,
        q: _QualifiedGoal,
    ) -> R16OwnerOutcome:
        target = q.target
        goal_id = goal_id_for_target(target)
        evidence = q.evidence
        origin_refs = q.origin_refs
        assert event.processing_sequence is not None
        proposal_digest = goal_proposal_digest(
            target,
            _DESCRIPTION,
            origin_refs=origin_refs,
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
            origin_refs=origin_refs,
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
            event, action, selection, q, goal_id, loop._goal_system.snapshot()
        )
        return R16OwnerOutcome(R16OwnerStatus.PROPOSED, goal_id)

    def _review(self, loop: SuzkaMainLoop, nonce: object) -> R16OwnerOutcome:
        event = self._current_event(loop, nonce, "review")
        assert self._inflight is not None
        goal_id = self._inflight[2]
        if type(goal_id) is not str:
            raise _GoalSourceUnavailable("invalid Goal review selection")
        goal = loop._goal_system.get(goal_id)
        before = loop._goal_system.snapshot()
        if goal is None or goal.lifecycle is not GoalLifecycle.PROPOSED:
            self._pending = _PendingGoal(event, "hold", None, None, None, before)
            return R16OwnerOutcome.hold(R16OwnerReason.NOT_PENDING_REVIEW)
        try:
            selection = self._selection_from_goal(goal)
            q = self._qualify_selection(selection, event, continuation=True)
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
                raise _GoalSourceUnavailable(
                    "an adopted Goal conflicts",
                    reason=R16OwnerReason.CONFLICTING_GOAL,
                )
        except _GoalSourceUnavailable as error:
            self._pending = _PendingGoal(event, "hold", None, None, goal_id, before)
            return R16OwnerOutcome.hold(owner_reason(error), goal_id)
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
        return R16OwnerOutcome(R16OwnerStatus.ADOPTED, goal_id)

    def _validate_commit(self, event: AgentEvent) -> None:
        pending = self._pending
        if event.event_type not in {
            AgentEventType.GOAL_PROPOSAL_REVIEW,
            AgentEventType.GOAL_SELF_PROPOSAL_REVIEW,
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
        if pending.action == "self_proposal":
            if type(pending.selection) is not _SelfGoalSelection:
                raise _GoalSourceUnavailable("self-Goal selection missing")
            fresh = self._qualify_self(pending.selection, event, proposal=True)
        elif pending.action == "proposal":
            if type(pending.selection) is not _GoalSourceSelection:
                raise _GoalSourceUnavailable("Goal selection missing")
            fresh = self._qualify(pending.selection, event)
        else:
            fresh = self._qualify_selection(
                pending.selection, event, continuation=True
            )
        if fresh != pending.qualification:
            raise _GoalSourceUnavailable("Goal source changed before WAL")
        goal = self._loop._goal_system.get(pending.goal_id or "")
        if goal is None:
            raise _GoalSourceUnavailable("Goal disappeared before WAL")
        self._expect_goal_contract(goal, fresh)
        if pending.action == "adopt":
            self._require_real_proposal_review(goal)
        if (
            pending.action in ("proposal", "self_proposal")
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
