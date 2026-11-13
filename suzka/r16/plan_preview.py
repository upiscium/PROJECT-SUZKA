"""R16 U3-A: source-qualified, non-authoritative Experience review Plan preview.

This private owner produces an immutable U1 Decision *value* for inspection,
not a durable Decision/issuer receipt, executable Action or public Explanation.
Step meanings are fixed predicates on actual R07/R12/R13/Value/Context facts;
caller/model-authored plan steps or confidence scores are never accepted.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Final

from suzka.identity.origin import (
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
    validate_origin_id,
)
from suzka.identity.value_system import (
    ValueSystemSnapshot,
    recompute_seed_contract_digest,
    value_state_digest,
)
from suzka.identifiers import validate_identifier
from suzka.motivation.commitment import (
    CommitmentAdmissionReason,
    CommitmentLifecycle,
    CommitmentRevisionOperation,
)
from suzka.motivation.commitment_system import CommitmentSystemEventOperation
from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.goal import (
    GoalAdmissionReason,
    GoalLifecycle,
    GoalRevisionOperation,
    GoalRecord,
)
from suzka.motivation.goal_system import GoalSystemEventOperation
from suzka.r16.commitment_owner import (
    _CommitmentSourceOwner,
    _CommitmentSourceUnavailable,
)
from suzka.r16.contracts import (
    MAX_JSON_BYTES,
    SCHEMA_VERSION,
    UNCERTAINTY_SCALE,
    CandidateKind,
    CandidateSelection,
    DecisionRecord,
    GoalEvidenceRef,
    HoldAssessment,
    PlanCandidate,
    PlanDAG,
    PlanStep,
    ReasonCode,
    StepKind,
    ValueEvidenceRef,
)
from suzka.r16.goal_owner import (
    _GoalSourceOwner,
    _GoalSourceUnavailable,
    _QualifiedGoal,
)
from suzka.r16.source_owner import _MotivationSourceOwner
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.context import ContextStatus
from suzka.runtime.event_journal import EventJournal, EventLifecycle
from suzka.runtime.journal_source import (
    JournalSourceReason,
    JournalSourceStage,
    SourceUnavailable,
    reason_of as _reason_of,
    locate_journal_source,
)

if TYPE_CHECKING:
    from suzka.runtime.main_loop import SuzkaMainLoop

_POLICY_VERSION: Final = 1
_VALUE_ID: Final = "honesty"
_NORMATIVE_POLICY: Final = "human-approved-source-review-honesty/v1"
_PLAN_A: Final = "source_first"
_PLAN_B: Final = "responsibility_review"
_VERIFY: Final = "verify_goal_source"
_COMPARE: Final = "compare_review_constraints"
_ASSESS: Final = "assess_review_readiness"
_UNKNOWN_OUTCOME_UNCERTAINTY: Final = UNCERTAINTY_SCALE


class _PlanSourceUnavailable(SourceUnavailable):
    """No qualified preview; never use source text in errors."""


class _PlanSourceConflict(_PlanSourceUnavailable):
    """Explicit contradiction, not general lack of evidence."""


def _stable_id(*parts: str) -> str:
    """Private deterministic identity of a *preview value*, not an R13 issuer."""
    return sha256(
        b"PROJECT-SUZKA:R16:PREVIEW:V1\0" + "\x00".join(parts).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class _StepObservation:
    """A code-owned ontology operation with checked inputs, never an Action."""

    step_id: str
    kind: StepKind
    predicate: str
    references: tuple[str, ...]
    satisfied: bool


@dataclass(frozen=True, slots=True)
class _QualifiedPreview:
    goal_id: str
    goal_revision: int
    goal_digest: str
    goal_adoption_event_id: str
    experience_id: str
    experience_digest: str
    motivation_revision_digest: str
    context_id: str
    context_revision: int
    context_last_modified_revision: int
    value_id: str
    value_revision: int
    value_digest: str
    value_origin_id: str
    commitment_id: str | None
    commitment_revision: int | None
    commitment_digest: str | None
    commitment_admission_event_id: str | None

    @property
    def qualified_commitment(self) -> bool:
        return self.commitment_id is not None


@dataclass(frozen=True, slots=True)
class _PlanPolicyWitness:
    """Private audit of a deterministic branch, not an R16 Decision receipt."""

    policy_version: int
    norm: str
    qualification: _QualifiedPreview
    selected_plan: str
    no_op_eligible: bool
    observations: tuple[_StepObservation, ...]


@dataclass(frozen=True, slots=True)
class _PlanPreviewResult:
    """One non-authoritative U1 value or explicit HOLD; no durable state."""

    decision: DecisionRecord | None
    hold: HoldAssessment | None
    witness: _PlanPolicyWitness | None

    def __post_init__(self) -> None:
        if (self.decision is None) == (self.hold is None):
            raise ValueError("a preview must be either a decision value or HOLD")
        if (self.decision is None) != (self.witness is None):
            raise ValueError("only qualified positive preview may have policy witness")

    @property
    def is_authoritative(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class _PendingPreview:
    event: AgentEvent
    result: _PlanPreviewResult
    initial_goal: object
    initial_value: ValueSystemSnapshot
    initial_motivation: object
    initial_commitment: object
    initial_context: object


class _PlanPreviewOwner:
    """Cooperative single-runtime Plan review without Decision persistence."""

    def __init__(
        self,
        loop: SuzkaMainLoop,
        runtime: AgentRuntime,
        journal: EventJournal,
        source_owner: _MotivationSourceOwner,
        goal_owner: _GoalSourceOwner,
        commitment_owner: _CommitmentSourceOwner,
    ) -> None:
        from suzka.runtime.main_loop import SuzkaMainLoop

        if (
            type(loop) is not SuzkaMainLoop
            or type(runtime) is not AgentRuntime
            or type(journal) is not EventJournal
            or type(source_owner) is not _MotivationSourceOwner
            or type(goal_owner) is not _GoalSourceOwner
            or type(commitment_owner) is not _CommitmentSourceOwner
            or not journal.has_exclusive_authority
            or loop._runtime is not runtime
            or loop._motivation_source_owner is not source_owner
            or loop._goal_source_owner is not goal_owner
            or loop._commitment_source_owner is not commitment_owner
            or goal_owner._source is not source_owner
            or commitment_owner._source is not source_owner
            or source_owner._journal is not journal
        ):
            raise _PlanSourceUnavailable("unqualified Plan preview owner binding")
        self._loop = loop
        self._runtime = runtime
        self._journal = journal
        self._source = source_owner
        self._goal = goal_owner
        self._commitment = commitment_owner
        self._inflight: tuple[object, str] | None = None
        self._pending: _PendingPreview | None = None
        loop._bind_plan_preview_owner(self)

    def _check_binding(self) -> None:
        if (
            self._loop._runtime is not self._runtime
            or self._loop._plan_preview_owner is not self
            or self._loop._motivation_source_owner is not self._source
            or self._loop._goal_source_owner is not self._goal
            or self._loop._commitment_source_owner is not self._commitment
        ):
            raise _PlanSourceUnavailable("Plan preview authority changed")
        self._goal._check_binding()
        self._commitment._check_binding()

    def _submit_preview(
        self, goal_id: str
    ) -> Future[AgentEventOutcome[_PlanPreviewResult]]:
        self._check_binding()
        validate_identifier(goal_id)
        nonce = object()

        def handle() -> _PlanPreviewResult:
            if self._inflight is not None or self._pending is not None:
                raise _PlanSourceUnavailable("Plan preview already pending")
            self._inflight = (nonce, goal_id)
            try:
                return self._loop._review_plan_preview(self, nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(
            AgentEventType.PLAN_PREVIEW_REVIEW,
            AgentEventSource.RUNTIME_PLAN_PREVIEW_REVIEW,
            handle,
        )

    def _current_event(self, loop: SuzkaMainLoop, nonce: object) -> AgentEvent:
        self._check_binding()
        event = self._runtime.current_event()
        if (
            loop is not self._loop
            or self._inflight is None
            or self._inflight[0] is not nonce
            or event is None
            or event.event_type is not AgentEventType.PLAN_PREVIEW_REVIEW
            or event.source is not AgentEventSource.RUNTIME_PLAN_PREVIEW_REVIEW
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _PlanSourceUnavailable("unbound Plan preview event")
        return event

    def _require_terminal(
        self,
        event_id: str,
        event_sequence: int,
        event_type: AgentEventType,
        event_source: AgentEventSource,
    ) -> None:
        """Require an earlier owner event to be terminal (IP-4c: three-stage).

        Callers check the owner's committed receipt for the event first, so
        an archived event is accepted on that receipt; a missing one is not.
        """
        inspection = self._journal.inspect()
        if (
            inspection.external_reconciliation_required
            or inspection.open_transactions
            or inspection.reconciliation_required_transactions
        ):
            raise _PlanSourceUnavailable("Journal not reconciled")
        stage, rows = locate_journal_source(inspection, event_id, event_sequence)
        if stage is JournalSourceStage.MISSING:
            raise _PlanSourceUnavailable(
                "owner event provenance missing",
                reason=JournalSourceReason.SOURCE_MISSING,
            )
        for lifecycle in (
            ()
            if stage is JournalSourceStage.ARCHIVED
            else (EventLifecycle.ACCEPTED, EventLifecycle.COMPLETED)
        ):
            matches = tuple(row for row in rows if row.lifecycle is lifecycle)
            if (
                len(matches) != 1
                or matches[0].event_type is not event_type
                or matches[0].source is not event_source
                or (
                    lifecycle is EventLifecycle.COMPLETED
                    and matches[0].processing_sequence != event_sequence
                )
            ):
                raise _PlanSourceUnavailable("owner event provenance not terminal")
        if any(row.lifecycle is EventLifecycle.FAILED for row in rows):
            raise _PlanSourceUnavailable("prior owner event failed")

    def _require_adopted_goal(
        self, goal: GoalRecord, event: AgentEvent
    ) -> _QualifiedGoal:
        if (
            goal.lifecycle is not GoalLifecycle.ADOPTED
            or goal.subject_admission is None
            or goal.subject_admission.reason is not GoalAdmissionReason.SUBJECT_REVIEW
            or goal.revision != 1
            or len(goal.revision_history) != 2
            or goal.revision_history[-1].operation is not GoalRevisionOperation.ADOPT
        ):
            raise _PlanSourceUnavailable("Goal is not first-review adopted")
        self._goal._require_real_proposal_review(goal)
        selection = self._goal._selection_from_goal(goal)
        qualified = self._goal._qualify(selection, event, continuation=True)
        self._goal._expect_goal_contract(goal, qualified)

        admission = goal.subject_admission
        revision = goal.revision_history[-1]
        receipts = tuple(
            r
            for r in self._loop._goal_system.snapshot().event_receipts
            if r.operation is GoalSystemEventOperation.ADOPT
            and r.goal_id == goal.goal_id
        )
        if len(receipts) != 1:
            raise _PlanSourceUnavailable("Goal admission receipt missing")
        receipt = receipts[0]
        if (
            receipt.event_id != admission.event_id
            or receipt.event_sequence != admission.event_sequence
            or receipt.proposal_digest != goal.proposal_digest
            or receipt.admission_digest != admission.admission_digest
            or receipt.evidence_refs != goal.evidence_refs
            or receipt.revision_witness != revision
            or revision.event_id != admission.event_id
            or revision.event_sequence != admission.event_sequence
            or event.processing_sequence is None
            or receipt.event_sequence >= event.processing_sequence
        ):
            raise _PlanSourceUnavailable("Goal admission provenance mismatched")
        self._require_terminal(
            admission.event_id,
            admission.event_sequence,
            AgentEventType.GOAL_ADOPTION_REVIEW,
            AgentEventSource.RUNTIME_GOAL_ADOPTION_REVIEW,
        )
        return qualified

    def _qualify_value(self, context_id: str) -> tuple[int, str, str]:
        values = self._loop._value_system.snapshot()
        matches = tuple(value for value in values.values if value.value_id == _VALUE_ID)
        if len(matches) != 1:
            raise _PlanSourceUnavailable("review integrity Value missing")
        value = matches[0]
        if (
            not value.is_active()
            or not value.applies_to(context_id)
            or value.origin.actor is not OriginActor.SYSTEM
            or value.origin.input_kind is not OriginInputKind.CONFIG_SEED
            or value.origin.admission is not ValueAdmissionStatus.SYSTEM_AUTHORIZED
            or value.origin.source_ref != f"config-seed:{_VALUE_ID}"
            or value.polarity != 1
            or value.seed_contract_digest is None
        ):
            raise _PlanSourceUnavailable("review integrity Value not qualified")
        seed = tuple(
            item.to_declaration()
            for item in self._loop.settings.values.seeds
            if item.value_id == _VALUE_ID
        )
        if (
            len(seed) != 1
            or recompute_seed_contract_digest(seed[0]) != value.seed_contract_digest
        ):
            raise _PlanSourceUnavailable("review Value has unqualified seed origin")
        return (
            value.revision,
            value_state_digest(value),
            validate_origin_id(value.origin),
        )

    def _qualify_commitment(
        self,
        target_id: str,
        context_id: str,
        motivation_revision_digest: str,
        event: AgentEvent,
    ) -> tuple[str, int, str, str] | None:
        x_ref = R13Reference(R13ReferenceKind.EXPERIENCE, target_id)
        candidates = tuple(
            record
            for record in self._loop._commitment_system.snapshot().records
            if x_ref in record.evidence_refs
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            raise _PlanSourceConflict("ambiguous responsibility for target")
        record = candidates[0]
        if (
            record.lifecycle is not CommitmentLifecycle.ACTIVE
            or record.revision != 1
            or len(record.revision_history) != 2
            or record.revision_history[-1].operation
            is not CommitmentRevisionOperation.ADMIT
            or record.subject_admission is None
            or record.subject_admission.reason
            is not CommitmentAdmissionReason.SUBJECT_RESPONSIBILITY
        ):
            raise _PlanSourceUnavailable("responsibility unaccepted or unavailable")
        selection = self._commitment._selection_from_record(record)
        q = self._commitment._qualify(selection, event, continuation=True)
        self._commitment._assert_contract(record, q)
        self._commitment._assert_original_proposal(record)
        if (
            q.context_id != context_id
            or q.target_experience_id != target_id
            or q.motivation_revision_digest != motivation_revision_digest
        ):
            raise _PlanSourceConflict("responsibility source differs from Goal")
        admission = record.subject_admission
        revision = record.revision_history[-1]
        receipts = tuple(
            r
            for r in self._loop._commitment_system.snapshot().event_receipts
            if r.operation is CommitmentSystemEventOperation.ACCEPT
            and r.commitment_id == record.commitment_id
        )
        if len(receipts) != 1:
            raise _PlanSourceUnavailable("responsibility admission receipt missing")
        receipt = receipts[0]
        if (
            receipt.event_id != admission.event_id
            or receipt.event_sequence != admission.event_sequence
            or receipt.admission_digest != admission.admission_digest
            or receipt.admission != admission
            or receipt.revision_witness != revision
            or receipt.evidence_refs != record.evidence_refs
            or event.processing_sequence is None
            or receipt.event_sequence >= event.processing_sequence
        ):
            raise _PlanSourceUnavailable("responsibility admission receipt mismatch")
        self._require_terminal(
            admission.event_id,
            admission.event_sequence,
            AgentEventType.COMMITMENT_ACCEPTANCE_REVIEW,
            AgentEventSource.RUNTIME_COMMITMENT_ACCEPTANCE_REVIEW,
        )
        return (
            record.commitment_id,
            record.revision,
            record.record_digest,
            admission.event_id,
        )

    def _qualify(self, goal_id: str, event: AgentEvent) -> _QualifiedPreview:
        self._check_binding()
        goal = self._loop._goal_system.get(goal_id)
        if goal is None:
            raise _PlanSourceUnavailable("no source-qualified Goal")
        try:
            q_goal = self._require_adopted_goal(goal, event)
            admission = goal.subject_admission
            if admission is None:
                raise _PlanSourceUnavailable("Goal admission lost")
            registry = self._loop.context_registry
            # Existing source owners check current ACTIVE Context and matching
            # channel/session. Bind its current generation only for this preview;
            # do not attest the historical Context revision at X creation.
            frame = registry.get(q_goal.context_id)
            context = registry.state
            if (
                context.current_context_id != frame.context_id
                or frame.status is not ContextStatus.ACTIVE
                or not any(x == frame for x in context.frames)
            ):
                raise _PlanSourceUnavailable("current Context not qualified")
            value_revision, value_digest, origin_id = self._qualify_value(
                frame.context_id
            )
            obligation = self._qualify_commitment(
                q_goal.target_experience_id,
                frame.context_id,
                q_goal.motivation_revision_digest,
                event,
            )
            return _QualifiedPreview(
                goal_id=goal.goal_id,
                goal_revision=goal.revision,
                goal_digest=goal.record_digest,
                goal_adoption_event_id=admission.event_id,
                experience_id=q_goal.target_experience_id,
                experience_digest=q_goal.target_record_digest,
                motivation_revision_digest=q_goal.motivation_revision_digest,
                context_id=frame.context_id,
                context_revision=context.revision,
                context_last_modified_revision=frame.last_modified_revision,
                value_id=_VALUE_ID,
                value_revision=value_revision,
                value_digest=value_digest,
                value_origin_id=origin_id,
                commitment_id=None if obligation is None else obligation[0],
                commitment_revision=None if obligation is None else obligation[1],
                commitment_digest=None if obligation is None else obligation[2],
                commitment_admission_event_id=(
                    None if obligation is None else obligation[3]
                ),
            )
        except _PlanSourceUnavailable:
            raise
        except (_GoalSourceUnavailable, _CommitmentSourceUnavailable) as exc:
            raise _PlanSourceUnavailable(
                "earlier source owner unqualified", reason=exc.reason
            ) from exc
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            raise _PlanSourceUnavailable(
                "Plan evidence unavailable", reason=_reason_of(exc)
            ) from exc

    @staticmethod
    def _plan(
        goal_ref: GoalEvidenceRef,
        value_ref: ValueEvidenceRef,
        policy_branch: str,
    ) -> PlanDAG:
        stamp = _stable_id(
            goal_ref.goal_id,
            goal_ref.goal_digest,
            value_ref.value_id,
            value_ref.value_digest,
            policy_branch,
        )
        steps: tuple[PlanStep, ...]
        if policy_branch == _PLAN_A:
            steps = (
                PlanStep(SCHEMA_VERSION, _VERIFY, StepKind.CHECK, ()),
                PlanStep(SCHEMA_VERSION, _ASSESS, StepKind.ASSESS, (_VERIFY,)),
            )
        else:
            steps = (
                PlanStep(SCHEMA_VERSION, _VERIFY, StepKind.CHECK, ()),
                PlanStep(SCHEMA_VERSION, _COMPARE, StepKind.COMPARE, (_VERIFY,)),
                PlanStep(SCHEMA_VERSION, _ASSESS, StepKind.ASSESS, (_COMPARE,)),
            )
        return PlanDAG(SCHEMA_VERSION, stamp, steps)

    @staticmethod
    def _observations(q: _QualifiedPreview) -> tuple[_StepObservation, ...]:
        # Execute bounded read-only ontology predicates, not only differences
        # in step names. The C presence observation is real R13 acceptance.
        return (
            _StepObservation(
                _VERIFY,
                StepKind.CHECK,
                "source_qualified_and_goal_adopted",
                (q.experience_id, q.goal_id, q.goal_adoption_event_id),
                bool(q.experience_digest and q.motivation_revision_digest),
            ),
            _StepObservation(
                _COMPARE,
                StepKind.COMPARE,
                "independent_responsibility_present_or_verified_absent",
                (q.value_id,)
                if q.commitment_id is None
                else (
                    q.value_id,
                    q.commitment_id,
                    q.commitment_admission_event_id or "",
                ),
                q.value_id == _VALUE_ID
                and (
                    q.commitment_id is None
                    or (
                        q.commitment_digest is not None
                        and q.commitment_admission_event_id is not None
                    )
                ),
            ),
            _StepObservation(
                _ASSESS,
                StepKind.ASSESS,
                "source_ready_with_normative_integrity_value",
                (q.goal_id, q.value_id, q.context_id),
                q.goal_revision == 1
                and q.value_revision >= 0
                and bool(q.value_digest and q.value_origin_id),
            ),
        )

    def _build_decision(self, q: _QualifiedPreview) -> _PlanPreviewResult:
        goal_ref = GoalEvidenceRef(
            SCHEMA_VERSION, q.goal_id, q.goal_revision, q.goal_digest
        )
        value_ref = ValueEvidenceRef(
            SCHEMA_VERSION, q.value_id, q.value_revision, q.value_digest
        )
        a = PlanCandidate(
            SCHEMA_VERSION,
            "candidate-a",
            CandidateKind.PLAN,
            goal_ref,
            value_ref,
            self._plan(goal_ref, value_ref, _PLAN_A),
        )
        b = PlanCandidate(
            SCHEMA_VERSION,
            "candidate-b",
            CandidateKind.PLAN,
            goal_ref,
            value_ref,
            self._plan(goal_ref, value_ref, _PLAN_B),
        )
        noop = PlanCandidate(
            SCHEMA_VERSION,
            "candidate-noop",
            CandidateKind.NO_OP,
            goal_ref,
            value_ref,
            None,
        )
        candidates = (a, b, noop)
        obs = self._observations(q)
        if not all(item.satisfied for item in obs):
            raise _PlanSourceUnavailable("Plan ontology predicate failed")
        branch = _PLAN_B if q.qualified_commitment else _PLAN_A
        selected = b if q.qualified_commitment else a
        identity = _stable_id(
            q.goal_id,
            q.goal_digest,
            q.value_digest,
            q.context_id,
            str(q.context_last_modified_revision),
            q.commitment_digest or "none",
            _NORMATIVE_POLICY,
        )
        decision = DecisionRecord(
            SCHEMA_VERSION,
            identity,
            candidates,
            CandidateSelection(
                SCHEMA_VERSION,
                selected.candidate_id,
                selected.content_digest,
            ),
            ReasonCode.PLAN_PREFERRED,
            _UNKNOWN_OUTCOME_UNCERTAINTY,
        )
        if len(decision.canonical_bytes()) > MAX_JSON_BYTES:
            raise _PlanSourceUnavailable("preview exceeds pure contract bound")
        witness = _PlanPolicyWitness(
            _POLICY_VERSION,
            _NORMATIVE_POLICY,
            q,
            branch,
            False,
            obs,
        )
        return _PlanPreviewResult(decision, None, witness)

    @staticmethod
    def _hold(reason: ReasonCode) -> _PlanPreviewResult:
        return _PlanPreviewResult(
            None, HoldAssessment(SCHEMA_VERSION, reason, UNCERTAINTY_SCALE), None
        )

    def _state_snapshot(
        self,
    ) -> tuple[object, ValueSystemSnapshot, object, object, object]:
        return (
            self._loop.export_goal_state(),
            self._loop._value_system.snapshot(),
            self._loop.export_motivation_state(),
            self._loop.export_commitment_state(),
            self._loop.context_registry.state,
        )

    def _preview(self, loop: SuzkaMainLoop, nonce: object) -> _PlanPreviewResult:
        event = self._current_event(loop, nonce)
        assert self._inflight is not None
        goal_id = self._inflight[1]
        initial = self._state_snapshot()
        try:
            before = self._qualify(goal_id, event)
            value = self._build_decision(before)
            # Catch current Context/Goal/Value/Commitment changes during preview
            # without claiming historical Context identity since source creation.
            after = self._qualify(goal_id, event)
            if after != before or self._state_snapshot() != initial:
                raise _PlanSourceUnavailable("Plan preview source changed")
        except _PlanSourceConflict:
            value = self._hold(ReasonCode.CONFLICTING_EVIDENCE)
        except _PlanSourceUnavailable:
            value = self._hold(ReasonCode.INSUFFICIENT_EVIDENCE)
        self._pending = _PendingPreview(event, value, *initial)
        return value

    def _validate_commit(self, event: AgentEvent) -> None:
        pending = self._pending
        if event.event_type is not AgentEventType.PLAN_PREVIEW_REVIEW:
            if pending is not None:
                raise _PlanSourceUnavailable("stale preview pending")
            return
        if pending is None or pending.event != event:
            raise _PlanSourceUnavailable("unbound preview pending")
        if self._state_snapshot() != (
            pending.initial_goal,
            pending.initial_value,
            pending.initial_motivation,
            pending.initial_commitment,
            pending.initial_context,
        ):
            raise _PlanSourceUnavailable("preview state changed before WAL")
        if pending.result.decision is None:
            return
        witness = pending.result.witness
        if witness is None:
            raise _PlanSourceUnavailable("qualified Plan evidence absent")
        fresh = self._qualify(witness.qualification.goal_id, event)
        if fresh != witness.qualification:
            raise _PlanSourceUnavailable("Plan review roots changed before WAL")
        if self._build_decision(fresh) != pending.result:
            raise _PlanSourceUnavailable("Plan policy changed before WAL")

    def _clear(self, event: AgentEvent) -> None:
        del event
        self._inflight = None
        self._pending = None
