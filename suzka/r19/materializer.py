"""R19 IP-20b: the Wake materializer (IP-20 D1 r3 §3, HD-1 (a), HD-20-1 (a)).

``WakeMaterializerV1`` owns the event ``R19_WAKE_MATERIALIZE`` and is the only
writer of the ``wakeups`` table.  One event is one scan: it writes **at most
one** Wake (the HP-5 "1 item" budget) and returns ``COMMITTED``, ``NONE_LEFT``
or ``HOLD(reason)``.  Wakes are created *after* the cause event is terminal,
from adopted records only (D-R19 §8-1); no R17 / R18 operation kind or handler
changes (A-6).

The four causes (D1 §3-2): ``GOAL_ADMITTED`` (the Goal's immutable
``GoalSubjectAdmission``), ``COMMITMENT_ADMITTED`` (the Commitment's
``CommitmentSubjectAdmission``), ``FEEDBACK_ADMITTED`` (an adopted
``DecisionFeedbackV1``) and ``OUTCOME_VERIFIED_NO_FEEDBACK`` (an adopted
``OutcomeAssessmentV1`` whose state is the terminal "not eligible" verdict
``VERIFIED_ABSENT`` / ``UNAUTHORIZED_EFFECT``, never "no Feedback row yet").
A scan takes its candidates in cause order (oldest cause first) and picks the
first that qualifies.  A candidate that does not is **skipped with a closed,
logged reason** (``SkipReason``), so one broken Goal never blocks the others
and nothing is frozen from an input that cannot be computed (D-R19 §3).

Everything is decided before the transaction opens, at the frozen instant
``T = event.requested_at`` and the evaluating event's own sequence (A-14):
the sealed policy input comes from ``_DecisionIssuer._evaluate`` (A-5).  A
PREPARE exception fail-stops the runtime (IP-14b §2-6), so every collision,
slot and capacity condition of ``insert_prepared_record`` is checked here
first and a hit is a HOLD with no row.

Scope (HD-20-1 (a), A-10): no activation marker.  A cause qualifies only if its
event is in the current incarnation (its sequence is above the incarnation's
processing horizon, so old admission receipts never create Wakes after a true
rollback), its Journal record is retained, and it is no older than
``materialize_max_age_s`` (default 7 days, ``MAX_WAKE_TTL``).  The age bound
reads the wall clock (the cause's Journal record time against T): it limits the
blast radius, it is not an exact floor, and production needs a marker before
R19 is enabled on a store with history.

With ``enabled`` false the event HOLDs ``R19_DISABLED`` before anything is
read.  Only the IP-20d Scheduler submits ``submit_scan``; this owner imports no
driver and the driver imports no R19 (D1 §3-3).  The result is public-safe: a
status, ids and closed reasons, never a source text.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
import logging
from typing import Final

from suzka.motivation.commitment import CommitmentAdmissionReason
from suzka.motivation.goal import GoalAdmissionReason, GoalLifecycle
from suzka.r16.contracts import checksum
from suzka.r16.cycle_contracts import FrozenPredecessor
from suzka.r16.decision_issuer import (
    _CONFLICT_HEADS,
    NO_DEADLINE_UTC,
    _ceil_utc,
    _DecisionIssuer,
    _Evaluation,
    _IssuerHold,
    _Unqualified,
    lineage_scope_for,
)
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_codec import CommittedDecisionV1, SourceRole
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import (
    AdoptedRecord,
    HeadBlockReason,
    ReaderClassification,
    ReaderTransient,
    SubjectHistoryReader,
)
from suzka.r16.history_records import (
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreError
from suzka.r16.successor_policy import (
    CommitmentState,
    PolicyRef,
    PredecessorState,
    SuccessorPolicyInputV2,
    utc_timestamp,
)
from suzka.r17.contracts import DecisionFeedbackV1, OutcomeAssessmentV1, OutcomeState
from suzka.r18.contracts import (
    OUTCOME_REF_KIND,
    ActionIntentV1,
    GoalOrigin,
    Direction,
    ExecutionClaimV1,
    VerifiedOutcomeV1,
)
from suzka.r19.contracts import (
    ASSESSMENT_REF_KIND,
    COMMITMENT_ADMISSION_REF_KIND,
    COMMITMENT_REF_KIND,
    EVENT_REF_KIND,
    EXPERIENCE_REF_KIND,
    GOAL_ADMISSION_REF_KIND,
    MAX_WAKE_TTL,
    MOTIVATION_REF_KIND,
    WakeCause,
    WakeKind,
    WakeTiming,
    WakeV1,
    wake_window,
    wakeup_id,
)
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.event_journal import EventJournalInspection, TransactionKind
from suzka.runtime.journal_source import JournalSourceStage, locate_journal_source
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionCoordinator,
)

LOGGER = logging.getLogger(__name__)

_EVENT_TYPES: Final = frozenset({AgentEventType.R19_WAKE_MATERIALIZE})
_EVENT_REF_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKE-EVENT-REF:V1\0"
# The terminal R17 verdicts of "this Outcome is not eligible for positive
# Feedback" (#373 A1-8): the only Assessments that start OUTCOME_VERIFIED_NO_FEEDBACK.
_NO_FEEDBACK_STATES: Final = frozenset(
    {OutcomeState.VERIFIED_ABSENT, OutcomeState.UNAUTHORIZED_EFFECT}
)
_OUTCOME_CAUSES: Final = frozenset(
    {WakeCause.FEEDBACK_ADMITTED, WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK}
)
# Per cause kind: the order of two causes with one sequence (never in practice).
_CAUSE_RANK: Final = {
    WakeCause.GOAL_ADMITTED: 0,
    WakeCause.COMMITMENT_ADMITTED: 1,
    WakeCause.FEEDBACK_ADMITTED: 2,
    WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK: 3,
}


class MaterializeStatus(str, Enum):
    COMMITTED = "wake_committed"
    NONE_LEFT = "wake_none_left"
    HOLD = "hold"


class MaterializeReason(str, Enum):
    """Why the scan held; a closed, source-free vocabulary."""

    R19_DISABLED = "r19_disabled"
    # Transient: the Journal is not reconciled, or a store / state read failed.
    HISTORY_UNAVAILABLE = "history_unavailable"
    HISTORY_CONFLICT = "history_conflict"
    # Transient: a key of the Wake is reserved but not adopted (PREPARED or
    # FINALIZED without its event terminal).
    IN_FLIGHT = "in_flight"
    HISTORY_CAPACITY = "history_capacity"


# The reasons the Scheduler retries with its bounded backoff.
TRANSIENT_MATERIALIZE_REASONS: Final = frozenset(
    {
        MaterializeReason.HISTORY_UNAVAILABLE,
        MaterializeReason.IN_FLIGHT,
        MaterializeReason.HISTORY_CAPACITY,
    }
)


class SkipReason(str, Enum):
    """Why one candidate made no Wake (D1 §3-1 step 5-6, §3-2); logged by the
    scan, never a durable row (A-8).  A skip never ends a retry: the next scan
    reads the candidate again."""

    # The sources, predecessor, lineage or interpretation are not qualified
    # now, or the ledger is TARGET_UNAVAILABLE (A-2): never frozen.
    NOT_QUALIFIED = "not_qualified"
    # (a) the Goal / Commitment of the cause is no longer the qualified one.
    STALE_CAUSE = "stale_cause"
    # (b) a Wake of the same Goal with a later cause exists.
    LATER_CAUSE_SCHEDULED = "later_cause_scheduled"
    # (c) the same evidence already has its Wake (``wake:<wakeup_id>``).
    SAME_FINGERPRINT = "same_fingerprint"
    # (d) ``wake_window`` is None: the deadline would be lost.
    WAKE_NOT_SCHEDULABLE = "wake_not_schedulable"
    # ``wake_window`` raised (a cause after T): never into the runtime.
    NOT_SCHEDULABLE = "not_schedulable"
    # Outside the current incarnation, past retention, or over-age (HD-20-1 (a)).
    OUT_OF_SCOPE = "out_of_scope"
    # One Wake per cause event / per verified Outcome revision (the keys).
    CAUSE_ALREADY_WOKEN = "cause_already_woken"
    OUTCOME_ALREADY_WOKEN = "outcome_already_woken"
    # The Feedback / Assessment names an Outcome that has a newer revision.
    OUTCOME_SUPERSEDED = "outcome_superseded"
    # The Outcome does not reach the Goal through the adopted chain.
    BINDING_MISMATCH = "binding_mismatch"
    # A value could not be built (a closed ValueError / TypeError): logged.
    INVALID_WAKE = "invalid_wake"


@dataclass(frozen=True, slots=True)
class _MaterializeResult:
    """The R19_WAKE_MATERIALIZE value (public-safe ids and closed reasons)."""

    status: MaterializeStatus
    wakeup_id: str | None
    reason: MaterializeReason | None
    wake: WakeV1 | None = None
    # The candidates this scan skipped before it ended: (cause, reason).
    skipped: tuple[tuple[WakeCause, SkipReason], ...] = field(default=())

    def __post_init__(self) -> None:
        if type(self.status) is not MaterializeStatus:
            raise TypeError("materialize status must be exact")
        if (self.status is MaterializeStatus.HOLD) != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not MaterializeReason:
            raise TypeError("reason must be an exact MaterializeReason")
        carries = self.status is MaterializeStatus.COMMITTED
        if carries != (self.wake is not None) or carries != (self.wakeup_id is not None):
            raise ValueError("only a committed result carries a Wake")
        if self.wake is not None and self.wake.wakeup_id != self.wakeup_id:
            raise ValueError("the result's id is not its Wake's")


@dataclass(frozen=True, slots=True)
class _Pending:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1


class _Hold(Exception):
    def __init__(self, reason: MaterializeReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _Skip(Exception):
    def __init__(self, reason: SkipReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _WakeMaterializeRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One cause that may make a Wake (cause order: sequence, rank, id)."""

    cause: WakeCause
    cause_event_id: str
    cause_sequence: int
    cause_ref: PolicyRef
    # GOAL / COMMITMENT: the Goal itself.  Outcome causes: None, resolved
    # through the adopted chain (``_goal_of``).
    goal_id: str | None = None
    commitment_id: str | None = None
    # Outcome causes: the Feedback / Assessment and the latest Outcome it names.
    cause_record: DecisionFeedbackV1 | OutcomeAssessmentV1 | None = None
    outcome: VerifiedOutcomeV1 | None = None

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.cause_sequence, _CAUSE_RANK[self.cause], self.cause_event_id)


@dataclass(frozen=True, slots=True)
class _GoalBinding:
    """The Goal an Outcome reaches: the Decision's Goal ref and lineage."""

    goal_id: str
    goal_revision: int
    goal_digest: str
    lineage_scope: str


@dataclass(slots=True)
class _Scan:
    """What one scan reads once: the classification, the adopted Wakes and the
    sealed inputs already built for a Goal (one evaluation per Goal)."""

    event: AgentEvent
    classified: ReaderClassification
    inspection: EventJournalInspection
    records: tuple[AdoptedRecord, ...]
    created_utc: str
    wakes: list[WakeV1]
    evaluations: dict[str, _Evaluation | _Unqualified | None] = field(default_factory=dict)
    skipped: list[tuple[WakeCause, SkipReason]] = field(default_factory=list)


def _valid_scan_event(event: AgentEvent | None) -> bool:
    return (
        event is not None
        and event.event_type in _EVENT_TYPES
        and event.source is AgentEventSource.RUNTIME_R19_WAKE_MATERIALIZE
        and type(event.processing_sequence) is int
        and event.processing_sequence >= 1
    )


class WakeMaterializerV1:
    """The R19 Wake materializer (the four causes of D-R19 §3)."""

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        timing: WakeTiming | None = None,
        max_age_s: int = int(MAX_WAKE_TTL.total_seconds()),
    ) -> None:
        if (
            type(decision_issuer) is not _DecisionIssuer
            or type(runtime) is not AgentRuntime
            or decision_issuer._preview._runtime is not runtime
            or type(enabled) is not bool
            or (timing is not None and type(timing) is not WakeTiming)
            or type(max_age_s) is not int
            or not 1 <= max_age_s <= int(MAX_WAKE_TTL.total_seconds())
        ):
            raise TypeError("unqualified wake materializer binding")
        self._decision = decision_issuer
        self._runtime = runtime
        self._store = decision_issuer._store
        self._journal = decision_issuer._journal
        self._recovery = decision_issuer._recovery
        self._enabled = enabled
        self._timing = timing or WakeTiming()
        self._max_age = timedelta(seconds=max_age_s)
        self._inflight: object | None = None
        self._pending: _Pending | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------ submission

    def submit_scan(self) -> Future[AgentEventOutcome[object]]:
        """One materialize scan (the IP-20d Scheduler is its only submitter)."""

        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise _WakeMaterializeRefused("wake materialize already pending")
            self._inflight = nonce
            try:
                return self._scan(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(
            AgentEventType.R19_WAKE_MATERIALIZE,
            AgentEventSource.RUNTIME_R19_WAKE_MATERIALIZE,
            handle,
        )

    def _current_event(self, nonce: object) -> AgentEvent:
        event = self._runtime.current_event()
        if self._inflight is not nonce or not _valid_scan_event(event):
            raise _WakeMaterializeRefused("unbound wake materialize event")
        assert event is not None
        return event

    # ------------------------------------------------------------ handler

    def _scan(self, nonce: object) -> object:
        self._pending = None
        event = self._current_event(nonce)
        try:
            return self._scan_checked(event)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(MaterializeReason.HISTORY_UNAVAILABLE)
        except _Hold as hold:
            return self._hold(hold.reason)

    @staticmethod
    def _hold(reason: MaterializeReason) -> _MaterializeResult:
        LOGGER.info("R19 materializer result=hold reason=%s", reason.value)
        return _MaterializeResult(MaterializeStatus.HOLD, None, reason)

    def _scan_checked(self, event: AgentEvent) -> object:
        if not self._enabled:
            raise _Hold(MaterializeReason.R19_DISABLED)
        scan = self._open_scan(event)
        for candidate in sorted(self._candidates(scan), key=lambda item: item.sort_key):
            try:
                operation = self._operation(scan, candidate)
            except _Skip as skip:
                self._skip(scan, candidate, skip.reason)
                continue
            return self._prepare(scan, operation)
        LOGGER.info("R19 materializer result=none_left skipped=%d", len(scan.skipped))
        return _MaterializeResult(
            MaterializeStatus.NONE_LEFT, None, None, skipped=tuple(scan.skipped)
        )

    @staticmethod
    def _skip(scan: _Scan, candidate: _Candidate, reason: SkipReason) -> None:
        LOGGER.info(
            "R19 materializer scan skipped cause=%s reason=%s",
            candidate.cause.value,
            reason.value,
        )
        scan.skipped.append((candidate.cause, reason))

    # ------------------------------------------------------------ the prelude

    def _open_scan(self, event: AgentEvent) -> _Scan:
        """D1 §3-1 steps 2-3: a reconciled Journal and an adopted, unblocked
        SubjectHistory in the current incarnation."""

        try:
            inspection = self._journal.inspect()
        except Exception:
            raise _Hold(MaterializeReason.HISTORY_UNAVAILABLE) from None
        if (
            inspection.external_reconciliation_required
            or inspection.open_transactions
            or inspection.reconciliation_required_transactions
        ):
            raise _Hold(MaterializeReason.HISTORY_UNAVAILABLE)
        try:
            classified = SubjectHistoryReader(self._journal, self._store).classify(
                inspection=inspection
            )
        except ReaderTransient:
            raise _Hold(MaterializeReason.HISTORY_UNAVAILABLE) from None
        if classified.global_block is not None:
            raise _Hold(_blocked_reason(classified.global_block))
        wakes = [item.record for item in classified.records if type(item.record) is WakeV1]
        return _Scan(
            event=event,
            classified=classified,
            inspection=inspection,
            records=classified.records,
            created_utc=utc_timestamp(event.requested_at),
            wakes=wakes,
        )

    # ------------------------------------------------------------ candidates

    def _candidates(self, scan: _Scan) -> list[_Candidate]:
        found = self._admission_candidates()
        found.extend(self._outcome_candidates(scan))
        return found

    def _admission_candidates(self) -> list[_Candidate]:
        """GOAL_ADMITTED and COMMITMENT_ADMITTED, from the admissions in
        AgentState (read here, on the runtime thread: D1 §3-3)."""

        loop = self._decision._preview._loop
        commitments = tuple(loop._commitment_system.records)
        found: list[_Candidate] = []
        for goal in loop._goal_system.records:
            admission = goal.subject_admission
            if (
                goal.lifecycle is not GoalLifecycle.ADOPTED
                or admission is None
                or admission.reason is not GoalAdmissionReason.SUBJECT_REVIEW
                or goal.revision != 1
            ):
                continue
            found.append(
                _Candidate(
                    WakeCause.GOAL_ADMITTED,
                    admission.event_id,
                    admission.event_sequence,
                    PolicyRef(
                        GOAL_ADMISSION_REF_KIND,
                        goal.goal_id,
                        goal.revision,
                        admission.admission_digest,
                    ),
                    goal_id=goal.goal_id,
                )
            )
            for commitment in commitments:
                cadmission = commitment.subject_admission
                if (
                    cadmission is None
                    or cadmission.reason is not CommitmentAdmissionReason.SUBJECT_RESPONSIBILITY
                    or goal.target not in commitment.evidence_refs
                ):
                    continue
                found.append(
                    _Candidate(
                        WakeCause.COMMITMENT_ADMITTED,
                        cadmission.event_id,
                        cadmission.event_sequence,
                        PolicyRef(
                            COMMITMENT_ADMISSION_REF_KIND,
                            commitment.commitment_id,
                            commitment.revision,
                            cadmission.admission_digest,
                        ),
                        goal_id=goal.goal_id,
                        commitment_id=commitment.commitment_id,
                    )
                )
        return found

    @staticmethod
    def _outcome_candidates(scan: _Scan) -> list[_Candidate]:
        """FEEDBACK_ADMITTED from adopted Feedback rows (no Wake before a
        qualified Feedback exists, #370 A1-4) and OUTCOME_VERIFIED_NO_FEEDBACK
        from adopted terminal Assessment verdicts only (#369 D2-A1)."""

        latest: dict[str, VerifiedOutcomeV1] = {}
        feedbacks: list[DecisionFeedbackV1] = []
        assessments: list[OutcomeAssessmentV1] = []
        for item in scan.records:
            record = item.record
            if type(record) is VerifiedOutcomeV1 and record.direction is Direction.FORWARD:
                held = latest.get(record.verified_outcome_id)
                if held is None or record.outcome_revision > held.outcome_revision:
                    latest[record.verified_outcome_id] = record
            elif type(record) is DecisionFeedbackV1:
                feedbacks.append(record)
            elif type(record) is OutcomeAssessmentV1 and record.outcome_state in _NO_FEEDBACK_STATES:
                assessments.append(record)
        found: list[_Candidate] = []
        for feedback in feedbacks:
            found.append(
                _Candidate(
                    WakeCause.FEEDBACK_ADMITTED,
                    feedback.admitting_event_id,
                    feedback.admitting_sequence,
                    feedback.policy_ref(),
                    cause_record=feedback,
                    outcome=latest.get(feedback.verified_outcome_id),
                )
            )
        for assessment in assessments:
            found.append(
                _Candidate(
                    WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK,
                    assessment.admitting_event_id,
                    assessment.admitting_sequence,
                    PolicyRef(
                        ASSESSMENT_REF_KIND,
                        assessment.assessment_id,
                        1,
                        assessment.assessment_digest,
                    ),
                    cause_record=assessment,
                    outcome=latest.get(assessment.verified_outcome_id),
                )
            )
        return found

    # ------------------------------------------------------------ one candidate

    def _operation(self, scan: _Scan, candidate: _Candidate) -> HistoryRecordOperationV1:
        """D1 §3-1 steps 3-8 for one candidate, or a ``_Skip``."""

        try:
            return self._operation_checked(scan, candidate)
        except (ValueError, TypeError) as error:
            # A closed contract refusal (a bad window, an invalid ref): logged,
            # never raised into the runtime.
            LOGGER.info(
                "R19 materializer candidate invalid cause=%s error=%s",
                candidate.cause.value,
                type(error).__name__,
            )
            raise _Skip(SkipReason.INVALID_WAKE) from None

    def _operation_checked(
        self, scan: _Scan, candidate: _Candidate
    ) -> HistoryRecordOperationV1:
        event = scan.event
        incarnation = scan.classified.incarnation
        # One Wake per cause event: the key, read from the adopted Wakes.
        if any(wake.cause_event_id == candidate.cause_event_id for wake in scan.wakes):
            raise _Skip(SkipReason.CAUSE_ALREADY_WOKEN)
        cause_utc = self._scope(scan, candidate)
        binding = self._binding(scan, candidate)
        goal_id = candidate.goal_id if binding is None else binding.goal_id
        assert goal_id is not None
        # (b) a later-cause Wake of the same Goal (a read, not a key).
        if any(
            wake.goal_id == goal_id and wake.cause_sequence > candidate.cause_sequence
            for wake in scan.wakes
        ):
            raise _Skip(SkipReason.LATER_CAUSE_SCHEDULED)
        evaluated = self._evaluated(scan, goal_id)
        value = self._qualify(candidate, binding, evaluated)
        fingerprint = _fingerprint(value)
        wake_id = wakeup_id(incarnation, goal_id, WakeKind.GOAL_REEVALUATION, fingerprint)
        # (c) the same evidence already has its Wake: adopted, or live in the
        # raw key index (a PREPARED / FINALIZED one is invisible to the reader).
        if any(wake.wakeup_id == wake_id for wake in scan.wakes) or self._store.live_keys(
            (f"wake:{wake_id}",)
        ):
            raise _Skip(SkipReason.SAME_FINGERPRINT)
        assert isinstance(evaluated, _Evaluation)
        wake = self._wake(scan, candidate, binding, evaluated, fingerprint, cause_utc)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R19_WAKE_MATERIALIZE,
            incarnation,
            event.event_id,
            event.processing_sequence or 0,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.WAKE,
            wake,
        )
        self._check_prepare(operation)
        return operation

    def _scope(self, scan: _Scan, candidate: _Candidate) -> str:
        """D1 §3-1 step 3 / HD-20-1 (a): the cause is in the current
        incarnation, retained, and not older than ``materialize_max_age_s``.
        Returns the cause time (the cause event's Journal record time)."""

        if candidate.cause_sequence <= scan.classified.horizon:
            raise _Skip(SkipReason.OUT_OF_SCOPE)
        stage, rows = locate_journal_source(
            scan.inspection,
            candidate.cause_event_id,
            candidate.cause_sequence,
        )
        if stage is not JournalSourceStage.RETAINED:
            raise _Skip(SkipReason.OUT_OF_SCOPE)
        caused = min(row.timestamp for row in rows)
        if scan.event.requested_at - caused > self._max_age:
            raise _Skip(SkipReason.OUT_OF_SCOPE)
        return utc_timestamp(caused)

    def _binding(self, scan: _Scan, candidate: _Candidate) -> _GoalBinding | None:
        """The Goal an Outcome cause reaches (D1 §3-2): Feedback / Assessment
        -> Outcome (still the latest revision) -> the adopted forward claim ->
        its Intent -> the adopted Decision -> its GOAL source ref.  None for the
        two admission causes."""

        if candidate.cause not in _OUTCOME_CAUSES:
            return None
        outcome = candidate.outcome
        cause = candidate.cause_record
        if outcome is None or cause is None:
            raise _Skip(SkipReason.BINDING_MISMATCH)
        if (
            outcome.outcome_revision != cause.outcome_revision
            or outcome.outcome_digest != cause.outcome_digest
        ):
            raise _Skip(SkipReason.OUTCOME_SUPERSEDED)
        key = (outcome.verified_outcome_id, outcome.outcome_revision)
        if any(wake.verified_outcome_key == key for wake in scan.wakes):
            raise _Skip(SkipReason.OUTCOME_ALREADY_WOKEN)
        claims = [
            item.record
            for item in scan.records
            if type(item.record) is ExecutionClaimV1
            and item.record.direction is Direction.FORWARD
            and item.record.operation_key == outcome.operation_key
        ]
        intents = (
            []
            if len(claims) != 1
            else [
                item.record
                for item in scan.records
                if type(item.record) is ActionIntentV1
                and item.record.intent_id == claims[0].intent_id
            ]
        )
        if len(intents) != 1:
            raise _Skip(SkipReason.BINDING_MISMATCH)
        intent = intents[0]
        for node in scan.classified.chains.get(intent.lineage_scope, ()):
            decision = node.decision
            if (
                decision.decision_id == intent.decision_id
                and decision.content_digest == intent.decision_committed_digest
            ):
                return _goal_binding(decision)
        raise _Skip(SkipReason.BINDING_MISMATCH)

    def _evaluated(self, scan: _Scan, goal_id: str) -> _Evaluation | _Unqualified:
        """D1 §3-1 step 4: the current policy input through the issuer, at the
        frozen T and the event's own sequence (A-5, A-14), once per Goal."""

        cached = scan.evaluations.get(goal_id)
        if cached is not None:
            return cached
        event = scan.event
        assert event.processing_sequence is not None
        try:
            result = self._decision._evaluate(
                event,
                goal_id,
                now=event.requested_at,
                before_sequence=event.processing_sequence,
            )
        except _IssuerHold:
            raise _Skip(SkipReason.NOT_QUALIFIED) from None
        scan.evaluations[goal_id] = result
        return result

    @staticmethod
    def _qualify(
        candidate: _Candidate,
        binding: _GoalBinding | None,
        evaluated: _Evaluation | _Unqualified,
    ) -> SuccessorPolicyInputV2:
        """The sealed input, once the cause is still the qualified one."""

        if isinstance(evaluated, _Unqualified):
            raise _Skip(SkipReason.NOT_QUALIFIED)
        q, value = evaluated.q, evaluated.value
        if candidate.cause is WakeCause.GOAL_ADMITTED:
            if q.goal_adoption_event_id != candidate.cause_event_id or q.goal_revision != 1:
                raise _Skip(SkipReason.STALE_CAUSE)
        elif candidate.cause is WakeCause.COMMITMENT_ADMITTED:
            if (
                q.commitment_id != candidate.commitment_id
                or q.commitment_admission_event_id != candidate.cause_event_id
                or value.commitment is not CommitmentState.ACTIVE
            ):
                raise _Skip(SkipReason.STALE_CAUSE)
        else:
            assert binding is not None
            if (q.goal_revision, q.goal_digest) != (binding.goal_revision, binding.goal_digest):
                raise _Skip(SkipReason.STALE_CAUSE)
            if lineage_scope_for(q.experience_id) != binding.lineage_scope:
                raise _Skip(SkipReason.BINDING_MISMATCH)
        return value

    def _wake(
        self,
        scan: _Scan,
        candidate: _Candidate,
        binding: _GoalBinding | None,
        evaluated: _Evaluation,
        fingerprint: str,
        cause_utc: str,
    ) -> WakeV1:
        """D1 §3-1 step 6 (d) and 8: the window, the frozen predecessor at the
        evaluation's own horizon, and the Wake."""

        q, value = evaluated.q, evaluated.value
        # NO_DEADLINE_UTC is the issuer's seal for "no deadline"; the Wake
        # field and ``wake_window`` take None for it (D1 §3-1).
        deadline = value.commitment_deadline_utc
        commitment_deadline = None if deadline in (None, NO_DEADLINE_UTC) else deadline
        goal = self._decision._preview._loop._goal_system.get(q.goal_id)
        goal_at = None if goal is None else goal.deadline.at
        try:
            window = wake_window(
                candidate.cause,
                cause_utc=cause_utc,
                created_utc=scan.created_utc,
                goal_deadline_utc=None if goal_at is None else _ceil_utc(goal_at),
                commitment_deadline_utc=commitment_deadline,
                timing=self._timing,
            )
        except ValueError:
            raise _Skip(SkipReason.NOT_SCHEDULABLE) from None
        if window is None:
            raise _Skip(SkipReason.WAKE_NOT_SCHEDULABLE)
        lineage = value.lineage_ref
        if (
            evaluated.incarnation != scan.classified.incarnation
            or lineage is None
            or evaluated.predecessor
            not in (PredecessorState.QUALIFIED, PredecessorState.ABSENT_INITIAL)
        ):
            raise _Skip(SkipReason.NOT_QUALIFIED)
        horizon = value.as_of_r07_sequence
        head = evaluated.head
        frozen = (
            FrozenPredecessor.absence(horizon)
            if evaluated.predecessor is PredecessorState.ABSENT_INITIAL
            else _frozen_head(head.decision if head is not None else None)
        )
        return WakeV1(
            kind=WakeKind.GOAL_REEVALUATION,
            cause=candidate.cause,
            cause_event_id=candidate.cause_event_id,
            cause_sequence=candidate.cause_sequence,
            cause_ref=candidate.cause_ref,
            goal_id=q.goal_id,
            goal_revision=q.goal_revision,
            goal_digest=q.goal_digest,
            experience_id=q.experience_id,
            goal_origin=GoalOrigin(q.goal_origin.name),
            origin_refs=_origin_refs(candidate, evaluated, binding),
            evidence_fingerprint=fingerprint,
            policy_kind=value.version,
            frozen_predecessor=frozen,
            horizon_sequence=horizon,
            incarnation_id=scan.classified.incarnation,
            journal_lineage_id=lineage.journal_lineage_id,
            not_before_utc=window[0],
            expires_at_utc=window[1],
            commitment_deadline_utc=commitment_deadline,
            context_id=q.context_id,
        )

    def _check_prepare(self, operation: HistoryRecordOperationV1) -> None:
        """Every condition under which ``insert_prepared_record`` would raise
        (a PREPARE exception fail-stops the runtime, IP-14b §2-6): the family
        switch, a live ``wake:`` / ``wake-cause:`` / ``wake-outcome:`` key
        (raw state, never ``adopted_records``) and FINALIZE headroom.  The
        ACTIVE phase is proven by the incarnation resolution of this scan."""

        if not self._store._r19_enabled:
            raise _Hold(MaterializeReason.R19_DISABLED)
        if not self._store.prepare_collisions(operation).clear:
            raise _Hold(MaterializeReason.IN_FLIGHT)
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _Hold(MaterializeReason.HISTORY_CAPACITY) from None

    # ------------------------------------------------------------ prepare

    def _prepare(self, scan: _Scan, operation: HistoryRecordOperationV1) -> object:
        event, incarnation = scan.event, scan.classified.incarnation
        wake = operation.record
        assert type(wake) is WakeV1
        snapshot_hash, _as_of = self._decision._snapshot()
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _Pending(
            event,
            incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(incarnation)),
            transaction_id,
            operation,
        )
        LOGGER.info("R19 materializer result=committed cause=%s", wake.cause.value)
        participant = SubjectHistoryRecordParticipant(
            self._store,
            incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(
            _MaterializeResult(
                MaterializeStatus.COMMITTED,
                wake.wakeup_id,
                None,
                wake,
                tuple(scan.skipped),
            ),
            (participant,),
        )

    # ------------------------------------------------------------ commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type not in _EVENT_TYPES:
            if pending is not None:
                raise _WakeMaterializeRefused("stale wake materialize pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise _WakeMaterializeRefused("unbound wake materialize pending")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _WakeMaterializeRefused("wake incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise _WakeMaterializeRefused("wake snapshot changed before WAL")
        rows = self._store.operations(pending.incarnation)
        own = [row for row in rows if row.transaction_id == pending.transaction_id]
        others = tuple(row for row in rows if row.transaction_id != pending.transaction_id)
        if (
            len(own) != 1
            or own[0].phase is not HistoryPhase.PREPARED
            or own[0].operation_kind is not pending.operation.operation_kind
            or own[0].pending_bytes != pending.operation.canonical_bytes()
            or own[0].operation_digest != pending.operation.operation_digest
            or operations_digest(others) != pending.rows_digest
        ):
            raise _WakeMaterializeRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None


# ------------------------------------------------------------------- values

def _blocked_reason(reason: HeadBlockReason | None) -> MaterializeReason:
    if reason in _CONFLICT_HEADS:
        return MaterializeReason.HISTORY_CONFLICT
    return MaterializeReason.HISTORY_UNAVAILABLE


def _fingerprint(value: SuccessorPolicyInputV2) -> str:
    """D1 §3-1 step 5; ``None`` (UNAVAILABLE source, predecessor, lineage or
    interpretation, or a TARGET_UNAVAILABLE ledger, A-2) is a NOT_QUALIFIED skip."""

    from suzka.r19.contracts import wake_fingerprint

    fingerprint = wake_fingerprint(value)
    if fingerprint is None:
        raise _Skip(SkipReason.NOT_QUALIFIED)
    return fingerprint


def _frozen_head(decision: CommittedDecisionV1 | None) -> FrozenPredecessor:
    if decision is None:
        raise _Skip(SkipReason.NOT_QUALIFIED)
    return FrozenPredecessor.decision(decision.decision_id, decision.content_digest)


def _goal_binding(decision: CommittedDecisionV1) -> _GoalBinding:
    goals = [ref for ref in decision.source_refs if ref.role is SourceRole.GOAL]
    if len(goals) != 1:
        raise _Skip(SkipReason.BINDING_MISMATCH)
    return _GoalBinding(goals[0].ref_id, goals[0].revision, goals[0].digest, decision.lineage_scope)


def _origin_refs(
    candidate: _Candidate, evaluated: _Evaluation, binding: _GoalBinding | None
) -> tuple[PolicyRef, ...]:
    """``origin_refs`` of D1 §3-2: empty for a user Goal, the self-origin Goal's
    MOTIVATION / EXPERIENCE / EVENT refs, the Commitment for its own cause, and
    the Outcome ref (plus the Goal's Commitment ref, P2-6) for the Outcome causes."""

    q, value = evaluated.q, evaluated.value
    refs: list[PolicyRef] = []
    if candidate.cause is WakeCause.GOAL_ADMITTED:
        if q.self_origin:
            refs = [
                PolicyRef(MOTIVATION_REF_KIND, q.motivation_id, 0, q.motivation_revision_digest),
                PolicyRef(EXPERIENCE_REF_KIND, q.experience_id, q.experience_revision, q.experience_digest),
                PolicyRef(
                    EVENT_REF_KIND,
                    q.experience_event_id,
                    q.experience_event_sequence,
                    checksum(_EVENT_REF_DOMAIN, [q.experience_event_id, q.experience_event_sequence]),
                ),
            ]
    elif candidate.cause is WakeCause.COMMITMENT_ADMITTED:
        refs = [_commitment_ref(value)]
    else:
        assert candidate.outcome is not None
        outcome = candidate.outcome
        refs = [
            PolicyRef(
                OUTCOME_REF_KIND,
                outcome.verified_outcome_id,
                outcome.outcome_revision,
                outcome.outcome_digest,
            )
        ]
        if value.commitment_ref is not None:
            refs.append(_commitment_ref(value))
    return tuple(sorted(refs, key=lambda item: item.sort_key))


def _commitment_ref(value: SuccessorPolicyInputV2) -> PolicyRef:
    ref = value.commitment_ref
    if ref is None or ref.kind != COMMITMENT_REF_KIND:
        raise _Skip(SkipReason.STALE_CAUSE)
    return ref


__all__ = [
    "TRANSIENT_MATERIALIZE_REASONS",
    "MaterializeReason",
    "MaterializeStatus",
    "SkipReason",
    "WakeMaterializerV1",
]
