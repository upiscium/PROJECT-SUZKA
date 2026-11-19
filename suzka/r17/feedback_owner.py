"""R17 IP-19: the Assessment and Feedback owner (#367 D1 / D2, D-R17 §3 / §4).

``FeedbackOwnerV1`` owns two events and is the only writer of
``outcome_assessments`` and ``decision_feedback`` (through the subject.history
record participant):

* ``R17_ASSESSMENT_COMMIT(outcome_id | scan)`` records R17's comparison and
  attribution for one terminal forward Outcome (every state: authorized,
  absent, unauthorized).  A VERIFIED_ABSENT or UNAUTHORIZED_EFFECT Assessment
  is R17's terminal verdict that the Outcome is not eligible for positive
  Feedback (IP-20's NO_FEEDBACK Wake starts from it);
* ``R17_FEEDBACK_COMMIT(assessment_id | scan)`` records the one positive
  Feedback an Assessment of an ``ACTIVE_VERIFIED_AUTHORIZED`` Outcome allows.

Both are installed beside the Decision issuer (the forecast issuer's shape).
While ``r18.enabled`` is false every event HOLDs ``R18_DISABLED`` before
anything is read and the driver never submits one.  Both events are ADMITTED
in the soft band (D-R17 §5, HP-8): they are cognition and create no obligation.

What makes a positive Feedback possible (the F09 / "no self-credit" rules):

1. the Outcome is an adopted forward Outcome written by the independent
   verifier (``OutcomeOwnerV1``), latest revision, state
   ``ACTIVE_VERIFIED_AUTHORIZED`` (so IN_DOUBT, VERIFIED_ABSENT and
   UNAUTHORIZED_EFFECT never reach it);
2. a forecast fixed before the claim exists (``forecast.admitting_sequence <
   claim.claim_sequence``), bound to the adopted Intent, challenge and claim;
3. the operation's ledger, derived from the adopted records, is exactly
   ``ACTIVE_VERIFIED_AUTHORIZED`` (an inverse already started, or a CORRUPT
   set, writes nothing new);
4. the attribution is ``assess()``'s: SELF only for an authorized effect whose
   effect row is this forward key with no foreign change; a foreign change is
   ENVIRONMENT, never SELF (R20-F09);
5. the record as it will project (the same ``read_feedback`` the Decision
   issuer uses, with this Feedback added) yields the recorded state.

Any miss is a HOLD with a closed reason and nothing is written.  Scans skip a
candidate that does not qualify (and log the reason) so one broken group never
blocks the others; the explicit-id form reports the reason.

``_validate_commit`` (after this event's own PREPARE) checks the incarnation,
the R06 snapshot hash and that SubjectHistory is exactly the evaluated rows
plus this transaction's PREPARED row; a failure is a runtime fail-stop,
recovered at startup (PREPARED -> tombstone, internally committed -> roll
forward).

Known limits (IP-19): the Wake that follows a Feedback or a NO_FEEDBACK
Assessment is IP-20's (HD-1 (a): R19 materializes it after this event is
terminal; the R17 operation kinds do not change).  V1 forecasts are nominal,
so a MATCH_SELF Feedback is not independent evidence that a prediction was
right (D-R17 §6).  The result is public-safe: a status, ids and a closed
reason, never request text.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Final, TypeVar

from suzka.r16.decision_issuer import _CONFLICT_HEADS, _DecisionIssuer, _sequence
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import (
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
from suzka.r16.successor_policy import FeedbackState, LedgerState
from suzka.r17.contracts import (
    DecisionFeedbackV1,
    ExpectedObservationV1,
    OutcomeAssessmentV1,
    assess,
)
from suzka.r17.feedback_read import read_feedback
from suzka.r18.contracts import (
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    VerifiedOutcomeV1,
)
from suzka.r18.ledger import RecordAt, derive_ledger
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.event_journal import TransactionKind
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionCoordinator,
)

LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")

# The driver's startup bound on consecutive scans of each kind.
MAX_FEEDBACK_SCANS_PER_STARTUP: Final = 16
_MAX_ID: Final = 256
_EVENT_TYPES: Final = frozenset(
    {AgentEventType.R17_ASSESSMENT_COMMIT, AgentEventType.R17_FEEDBACK_COMMIT}
)


class AssessmentCommitStatus(str, Enum):
    COMMITTED = "assessment_committed"
    EXISTING = "assessment_existing"
    NONE_LEFT = "assessment_none_left"
    HOLD = "hold"


class FeedbackCommitStatus(str, Enum):
    COMMITTED = "feedback_committed"
    EXISTING = "feedback_existing"
    NONE_LEFT = "feedback_none_left"
    HOLD = "hold"


class R17FeedbackReason(str, Enum):
    """Why the owner held; a closed, source-free vocabulary."""

    R18_DISABLED = "r18_disabled"
    HISTORY_UNAVAILABLE = "history_unavailable"
    HISTORY_CONFLICT = "history_conflict"
    OUTCOME_UNKNOWN = "outcome_unknown"
    ASSESSMENT_UNKNOWN = "assessment_unknown"
    # Transient: the Assessment exists in this incarnation but is not adopted.
    NOT_YET_TERMINAL = "not_yet_terminal"
    # No forecast for the Outcome's forward key: no Assessment, no Feedback
    # (the projection is INVALID and the cycle closes on a HOLD receipt).
    FORECAST_MISSING = "forecast_missing"
    BINDING_MISMATCH = "binding_mismatch"
    # The Assessment is not eligible for positive Feedback, a newer Outcome
    # revision exists, or the operation is no longer ACTIVE_VERIFIED_AUTHORIZED.
    NOT_ELIGIBLE = "not_eligible"
    # Transient: a key of this record is reserved but not adopted.
    IN_FLIGHT = "in_flight"
    HISTORY_CAPACITY = "history_capacity"


# The reasons the driver retries with its bounded delays.
TRANSIENT_FEEDBACK_REASONS: Final = frozenset(
    {
        R17FeedbackReason.NOT_YET_TERMINAL,
        R17FeedbackReason.IN_FLIGHT,
        R17FeedbackReason.HISTORY_UNAVAILABLE,
    }
)


@dataclass(frozen=True, slots=True)
class _AssessmentResult:
    """The R17_ASSESSMENT_COMMIT value (public-safe ids and a closed reason)."""

    status: AssessmentCommitStatus
    outcome_id: str | None
    assessment_id: str | None
    reason: R17FeedbackReason | None
    assessment: OutcomeAssessmentV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not AssessmentCommitStatus:
            raise TypeError("assessment status must be exact")
        if (self.status is AssessmentCommitStatus.HOLD) != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not R17FeedbackReason:
            raise TypeError("reason must be an exact R17FeedbackReason")
        carries = self.status in (
            AssessmentCommitStatus.COMMITTED,
            AssessmentCommitStatus.EXISTING,
        )
        if carries != (self.assessment is not None) or carries != (self.assessment_id is not None):
            raise ValueError("only a committed or existing result carries an Assessment")
        if self.assessment is not None and (
            self.assessment.assessment_id != self.assessment_id
            or self.assessment.verified_outcome_id != self.outcome_id
        ):
            raise ValueError("the result's ids are not its Assessment's")


@dataclass(frozen=True, slots=True)
class _FeedbackResult:
    """The R17_FEEDBACK_COMMIT value (public-safe ids and a closed reason)."""

    status: FeedbackCommitStatus
    assessment_id: str | None
    feedback_id: str | None
    reason: R17FeedbackReason | None
    feedback: DecisionFeedbackV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not FeedbackCommitStatus:
            raise TypeError("feedback status must be exact")
        if (self.status is FeedbackCommitStatus.HOLD) != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not R17FeedbackReason:
            raise TypeError("reason must be an exact R17FeedbackReason")
        carries = self.status in (FeedbackCommitStatus.COMMITTED, FeedbackCommitStatus.EXISTING)
        if carries != (self.feedback is not None) or carries != (self.feedback_id is not None):
            raise ValueError("only a committed or existing result carries a Feedback")
        if self.feedback is not None and (
            self.feedback.feedback_id != self.feedback_id
            or self.feedback.assessment_id != self.assessment_id
        ):
            raise ValueError("the result's ids are not its Feedback's")


@dataclass(frozen=True, slots=True)
class _Pending:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1


class _Hold(Exception):
    def __init__(self, reason: R17FeedbackReason, target_id: str | None = None) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.target_id = target_id


class _FeedbackCommitRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


@dataclass(frozen=True, slots=True)
class _Context:
    """The adopted chain behind one forward Outcome, from one classification."""

    forecast: ExpectedObservationV1
    intent: ActionIntentV1
    claim: ExecutionClaimV1
    ledger: LedgerState


def _valid_id(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= _MAX_ID


class FeedbackOwnerV1:
    """The R17 Assessment / Feedback owner for terminal forward Outcomes."""

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
    ) -> None:
        if (
            type(decision_issuer) is not _DecisionIssuer
            or type(runtime) is not AgentRuntime
            or decision_issuer._preview._runtime is not runtime
            or type(enabled) is not bool
        ):
            raise TypeError("unqualified feedback owner binding")
        self._decision = decision_issuer
        self._runtime = runtime
        self._store = decision_issuer._store
        self._journal = decision_issuer._journal
        self._recovery = decision_issuer._recovery
        self._clock: Callable[[], datetime] = decision_issuer._clock
        self._enabled = enabled
        self._inflight: tuple[object, AgentEventType, str | None] | None = None
        self._pending: _Pending | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------ submission

    def _submit_assessment(
        self, outcome_id: str | None
    ) -> Future[AgentEventOutcome[object]]:
        """``outcome_id`` names one forward Outcome; ``None`` is the scan."""

        if outcome_id is not None and not _valid_id(outcome_id):
            raise ValueError("an Assessment names one Outcome id, or scans")
        return self._submit(
            AgentEventType.R17_ASSESSMENT_COMMIT,
            AgentEventSource.RUNTIME_R17_ASSESSMENT_COMMIT,
            outcome_id,
        )

    def _submit_feedback(
        self, assessment_id: str | None
    ) -> Future[AgentEventOutcome[object]]:
        """``assessment_id`` names one Assessment; ``None`` is the scan."""

        if assessment_id is not None and not _valid_id(assessment_id):
            raise ValueError("a Feedback names one Assessment id, or scans")
        return self._submit(
            AgentEventType.R17_FEEDBACK_COMMIT,
            AgentEventSource.RUNTIME_R17_FEEDBACK_COMMIT,
            assessment_id,
        )

    def _submit(
        self,
        event_type: AgentEventType,
        source: AgentEventSource,
        target: str | None,
    ) -> Future[AgentEventOutcome[object]]:
        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise _FeedbackCommitRefused("feedback commit already pending")
            self._inflight = (nonce, event_type, target)
            try:
                return self._commit(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(event_type, source, handle)

    def _current_event(self, nonce: object) -> tuple[AgentEvent, AgentEventType, str | None]:
        event = self._runtime.current_event()
        inflight = self._inflight
        if (
            inflight is None
            or inflight[0] is not nonce
            or event is None
            or event.event_type is not inflight[1]
            or event.event_type not in _EVENT_TYPES
            or event.source
            is not (
                AgentEventSource.RUNTIME_R17_ASSESSMENT_COMMIT
                if event.event_type is AgentEventType.R17_ASSESSMENT_COMMIT
                else AgentEventSource.RUNTIME_R17_FEEDBACK_COMMIT
            )
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _FeedbackCommitRefused("unbound feedback commit event")
        return event, inflight[1], inflight[2]

    # ------------------------------------------------------------ handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event, event_type, target = self._current_event(nonce)
        assessing = event_type is AgentEventType.R17_ASSESSMENT_COMMIT
        try:
            if assessing:
                return self._assess_checked(event, target)
            return self._feedback_checked(event, target)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(assessing, R17FeedbackReason.HISTORY_UNAVAILABLE, target)
        except _Hold as hold:
            return self._hold(assessing, hold.reason, hold.target_id or target)

    @staticmethod
    def _hold(
        assessing: bool, reason: R17FeedbackReason, target: str | None
    ) -> _AssessmentResult | _FeedbackResult:
        LOGGER.info(
            "R17 %s owner result=hold reason=%s",
            "assessment" if assessing else "feedback",
            reason.value,
        )
        if assessing:
            return _AssessmentResult(AssessmentCommitStatus.HOLD, target, None, reason)
        return _FeedbackResult(FeedbackCommitStatus.HOLD, target, None, reason)

    # -------------------------------------------------------- the Assessment

    def _assess_checked(self, event: AgentEvent, outcome_id: str | None) -> object:
        if not self._enabled:
            raise _Hold(R17FeedbackReason.R18_DISABLED)
        classified = self._classify()
        records = _records(classified)
        outcomes = _latest_outcomes(records)
        assessments = _adopted(records, OutcomeAssessmentV1)
        if outcome_id is None:
            chosen = self._scan_assessment(event, classified, records, outcomes, assessments)
            if chosen is None:
                LOGGER.info("R17 assessment owner result=none_left")
                return _AssessmentResult(AssessmentCommitStatus.NONE_LEFT, None, None, None)
            outcome, assessment = chosen
        else:
            found = outcomes.get(outcome_id)
            if found is None:
                raise _Hold(R17FeedbackReason.OUTCOME_UNKNOWN, outcome_id)
            outcome = found[0]
            existing = _assessment_of(assessments, outcome)
            if existing is not None:
                return self._existing_assessment(existing)
            assessment = self._assessment(event, classified, records, outcome)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R17_ASSESSMENT_COMMIT,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.OUTCOME_ASSESSMENT,
            assessment,
        )
        return self._prepare(
            event,
            classified,
            operation,
            outcome.verified_outcome_id,
            lambda: _AssessmentResult(
                AssessmentCommitStatus.COMMITTED,
                assessment.verified_outcome_id,
                assessment.assessment_id,
                None,
                assessment,
            ),
        )

    @staticmethod
    def _existing_assessment(existing: OutcomeAssessmentV1) -> _AssessmentResult:
        LOGGER.info("R17 assessment owner result=existing")
        return _AssessmentResult(
            AssessmentCommitStatus.EXISTING,
            existing.verified_outcome_id,
            existing.assessment_id,
            None,
            existing,
        )

    def _scan_assessment(
        self,
        event: AgentEvent,
        classified: ReaderClassification,
        records: tuple[RecordAt, ...],
        outcomes: dict[str, tuple[VerifiedOutcomeV1, int]],
        assessments: list[tuple[OutcomeAssessmentV1, int]],
    ) -> tuple[VerifiedOutcomeV1, OutcomeAssessmentV1] | None:
        """The first adopted forward Outcome (processing order) whose latest
        revision has no Assessment and that qualifies; a candidate that does
        not is skipped, so one broken group never blocks the others."""

        for outcome, _sequence_of in sorted(outcomes.values(), key=lambda item: item[1]):
            if _assessment_of(assessments, outcome) is not None:
                continue
            try:
                return outcome, self._assessment(event, classified, records, outcome)
            except _Hold as hold:
                LOGGER.info(
                    "R17 assessment owner scan skipped reason=%s", hold.reason.value
                )
                if hold.reason in (R17FeedbackReason.IN_FLIGHT, R17FeedbackReason.HISTORY_CAPACITY):
                    raise
        return None

    def _assessment(
        self,
        event: AgentEvent,
        classified: ReaderClassification,
        records: tuple[RecordAt, ...],
        outcome: VerifiedOutcomeV1,
    ) -> OutcomeAssessmentV1:
        context = self._context(classified, records, outcome)
        try:
            assessment = OutcomeAssessmentV1.issue(
                context.forecast, outcome.to_r17_view(), event.event_id, _sequence(event)
            )
        except (TypeError, ValueError):
            raise _Hold(
                R17FeedbackReason.BINDING_MISMATCH, outcome.verified_outcome_id
            ) from None
        return assessment

    # ---------------------------------------------------------- the Feedback

    def _feedback_checked(self, event: AgentEvent, assessment_id: str | None) -> object:
        if not self._enabled:
            raise _Hold(R17FeedbackReason.R18_DISABLED)
        classified = self._classify()
        records = _records(classified)
        outcomes = _latest_outcomes(records)
        assessments = _adopted(records, OutcomeAssessmentV1)
        feedbacks = _adopted(records, DecisionFeedbackV1)
        if assessment_id is None:
            chosen = self._scan_feedback(event, classified, records, outcomes, assessments, feedbacks)
            if chosen is None:
                LOGGER.info("R17 feedback owner result=none_left")
                return _FeedbackResult(FeedbackCommitStatus.NONE_LEFT, None, None, None)
            feedback = chosen
        else:
            found = [item for item, _seq in assessments if item.assessment_id == assessment_id]
            if not found:
                self._assessment_missing(classified, assessment_id)
            assessment = found[0]
            existing = _feedback_of(feedbacks, assessment)
            if existing is not None:
                return self._existing_feedback(existing)
            feedback = self._feedback(event, classified, records, outcomes, assessment)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R17_FEEDBACK_COMMIT,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.DECISION_FEEDBACK,
            feedback,
        )
        return self._prepare(
            event,
            classified,
            operation,
            feedback.assessment_id,
            lambda: _FeedbackResult(
                FeedbackCommitStatus.COMMITTED,
                feedback.assessment_id,
                feedback.feedback_id,
                None,
                feedback,
            ),
        )

    @staticmethod
    def _existing_feedback(existing: DecisionFeedbackV1) -> _FeedbackResult:
        LOGGER.info("R17 feedback owner result=existing")
        return _FeedbackResult(
            FeedbackCommitStatus.EXISTING,
            existing.assessment_id,
            existing.feedback_id,
            None,
            existing,
        )

    def _assessment_missing(self, classified: ReaderClassification, assessment_id: str) -> None:
        """Not adopted: transient while its operation is still in the store."""

        for row in classified.rows:
            if (
                row.operation_kind is not HistoryOperationKind.R17_ASSESSMENT_COMMIT
                or row.phase not in (HistoryPhase.PREPARED, HistoryPhase.FINALIZED)
                or row.pending_bytes is None
            ):
                continue
            try:
                operation = HistoryRecordOperationV1.from_bytes(row.pending_bytes)
            except (TypeError, ValueError):
                raise _Hold(R17FeedbackReason.HISTORY_CONFLICT, assessment_id) from None
            record = operation.record
            if type(record) is OutcomeAssessmentV1 and record.assessment_id == assessment_id:
                raise _Hold(R17FeedbackReason.NOT_YET_TERMINAL, assessment_id)
        raise _Hold(R17FeedbackReason.ASSESSMENT_UNKNOWN, assessment_id)

    def _scan_feedback(
        self,
        event: AgentEvent,
        classified: ReaderClassification,
        records: tuple[RecordAt, ...],
        outcomes: dict[str, tuple[VerifiedOutcomeV1, int]],
        assessments: list[tuple[OutcomeAssessmentV1, int]],
        feedbacks: list[tuple[DecisionFeedbackV1, int]],
    ) -> DecisionFeedbackV1 | None:
        for assessment, _seq in sorted(assessments, key=lambda item: item[1]):
            if (
                not assessment.positive_feedback_eligible
                or _feedback_of(feedbacks, assessment) is not None
            ):
                continue
            try:
                return self._feedback(event, classified, records, outcomes, assessment)
            except _Hold as hold:
                LOGGER.info("R17 feedback owner scan skipped reason=%s", hold.reason.value)
                if hold.reason in (R17FeedbackReason.IN_FLIGHT, R17FeedbackReason.HISTORY_CAPACITY):
                    raise
        return None

    def _feedback(
        self,
        event: AgentEvent,
        classified: ReaderClassification,
        records: tuple[RecordAt, ...],
        outcomes: dict[str, tuple[VerifiedOutcomeV1, int]],
        assessment: OutcomeAssessmentV1,
    ) -> DecisionFeedbackV1:
        """The one positive Feedback of an Assessment, or a HOLD."""

        target = assessment.assessment_id
        if not assessment.positive_feedback_eligible:
            raise _Hold(R17FeedbackReason.NOT_ELIGIBLE, target)
        found = outcomes.get(assessment.verified_outcome_id)
        if found is None:
            raise _Hold(R17FeedbackReason.HISTORY_CONFLICT, target)
        outcome = found[0]
        if (
            outcome.outcome_revision != assessment.outcome_revision
            or outcome.outcome_digest != assessment.outcome_digest
        ):
            # A newer revision exists (or the digest differs): never reuse.
            raise _Hold(R17FeedbackReason.NOT_ELIGIBLE, target)
        context = self._context(classified, records, outcome)
        if context.ledger is not LedgerState.ACTIVE_VERIFIED_AUTHORIZED:
            raise _Hold(R17FeedbackReason.NOT_ELIGIBLE, target)
        try:
            expected = assess(context.forecast, outcome.to_r17_view())
            if (
                assessment.forecast_id != context.forecast.forecast_id
                or assessment.forecast_digest != context.forecast.forecast_digest
                or expected != (assessment.comparison, assessment.contributors)
            ):
                raise ValueError("the Assessment does not follow from its inputs")
            feedback = DecisionFeedbackV1.issue(assessment, event.event_id, _sequence(event))
        except (TypeError, ValueError):
            raise _Hold(R17FeedbackReason.BINDING_MISMATCH, target) from None
        # The record as it will project: the Decision issuer's own reader with
        # this Feedback added must yield the state recorded here.
        projected = read_feedback(
            (*records, RecordAt(feedback, _sequence(event))),
            context.intent.forward_key,
            context.ledger,
            _sequence(event) + 1,
        )
        if projected.state is not feedback.feedback_state or projected.ref != feedback.policy_ref():
            raise _Hold(R17FeedbackReason.BINDING_MISMATCH, target)
        if feedback.feedback_state is FeedbackState.ABSENT:  # pragma: no cover - defensive
            raise _Hold(R17FeedbackReason.BINDING_MISMATCH, target)
        return feedback

    # ------------------------------------------------------------ shared

    def _classify(self) -> ReaderClassification:
        reader = SubjectHistoryReader(self._journal, self._store)
        try:
            classified = reader.classify(inspection=self._journal.inspect())
        except ReaderTransient:
            raise _Hold(R17FeedbackReason.HISTORY_UNAVAILABLE) from None
        if classified.global_block is not None:
            raise _Hold(_blocked_reason(classified.global_block))
        return classified

    def _context(
        self,
        classified: ReaderClassification,
        records: tuple[RecordAt, ...],
        outcome: VerifiedOutcomeV1,
    ) -> _Context:
        """The adopted forecast -> Intent -> challenge -> claim chain of one
        forward Outcome (the ledger's checks, plus the forecast before the
        claim), and the operation's ledger state."""

        target = outcome.verified_outcome_id
        key = outcome.operation_key
        forecasts = [
            record
            for record, _seq in _adopted(records, ExpectedObservationV1)
            if record.forward_key == key
        ]
        if not forecasts:
            raise _Hold(R17FeedbackReason.FORECAST_MISSING, target)
        claims = [
            record
            for record, _seq in _adopted(records, ExecutionClaimV1)
            if record.direction is Direction.FORWARD and record.operation_key == key
        ]
        if len(forecasts) != 1 or len(claims) != 1:
            raise _Hold(R17FeedbackReason.HISTORY_CONFLICT, target)
        forecast, claim = forecasts[0], claims[0]
        intents = [
            record
            for record, _seq in _adopted(records, ActionIntentV1)
            if record.intent_id == claim.intent_id
        ]
        challenges = [
            record
            for record, _seq in _adopted(records, ApprovalChallengeV1)
            if record.challenge_digest == claim.challenge_digest
        ]
        if len(intents) != 1 or len(challenges) != 1:
            raise _Hold(R17FeedbackReason.HISTORY_CONFLICT, target)
        intent, challenge = intents[0], challenges[0]
        if (
            intent.forward_key != key
            or intent.intent_digest != claim.intent_digest
            or intent.incarnation_id != classified.incarnation
            or challenge.intent_id != intent.intent_id
            or challenge.nonce != claim.nonce
            or forecast.action_intent_id != intent.intent_id
            or forecast.action_intent_digest != intent.intent_digest
            or forecast.target_instance_id != outcome.target_instance_id
            # D-R17 §2: the forecast is fixed before the effect's claim.
            or forecast.admitting_sequence >= claim.claim_sequence
        ):
            raise _Hold(R17FeedbackReason.BINDING_MISMATCH, target)
        ledger = derive_ledger(records, intent.obligation_group_key, self._clock())
        if ledger.state is LedgerState.CORRUPT:
            raise _Hold(R17FeedbackReason.HISTORY_CONFLICT, target)
        return _Context(forecast, intent, claim, ledger.state)

    def _prepare(
        self,
        event: AgentEvent,
        classified: ReaderClassification,
        operation: HistoryRecordOperationV1,
        target: str,
        result: Callable[[], object],
    ) -> object:
        if self._store.live_keys(operation.reserved_keys):
            raise _Hold(R17FeedbackReason.IN_FLIGHT, target)
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _Hold(R17FeedbackReason.HISTORY_CAPACITY, target) from None
        snapshot_hash, _as_of = self._decision._snapshot()
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _Pending(
            event,
            classified.incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(classified.incarnation)),
            transaction_id,
            operation,
        )
        LOGGER.info(
            "R17 %s owner result=committed",
            "assessment"
            if operation.operation_kind is HistoryOperationKind.R17_ASSESSMENT_COMMIT
            else "feedback",
        )
        participant = SubjectHistoryRecordParticipant(
            self._store,
            classified.incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(result(), (participant,))

    # ------------------------------------------------------------ commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type not in _EVENT_TYPES:
            if pending is not None:
                raise _FeedbackCommitRefused("stale feedback commit pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise _FeedbackCommitRefused("unbound feedback commit pending")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _FeedbackCommitRefused("feedback incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise _FeedbackCommitRefused("feedback snapshot changed before WAL")
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
            raise _FeedbackCommitRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None


# ------------------------------------------------------------------- values


def _blocked_reason(reason: HeadBlockReason | None) -> R17FeedbackReason:
    if reason in _CONFLICT_HEADS:
        return R17FeedbackReason.HISTORY_CONFLICT
    return R17FeedbackReason.HISTORY_UNAVAILABLE


def _records(classified: ReaderClassification) -> tuple[RecordAt, ...]:
    return tuple(
        RecordAt(item.record, item.processing_sequence) for item in classified.records
    )


def _adopted(records: tuple[RecordAt, ...], kind: type[_T]) -> list[tuple[_T, int]]:
    """The adopted records of one exact type with their processing sequence,
    in processing order."""

    found: list[tuple[_T, int]] = []
    for item in records:
        if type(item.record) is kind:
            assert isinstance(item.record, kind)
            found.append((item.record, item.sequence))
    return sorted(found, key=lambda pair: pair[1])


def _latest_outcomes(
    records: tuple[RecordAt, ...],
) -> dict[str, tuple[VerifiedOutcomeV1, int]]:
    """Each forward Outcome id -> its latest adopted revision and sequence."""

    latest: dict[str, tuple[VerifiedOutcomeV1, int]] = {}
    for record, sequence in _adopted(records, VerifiedOutcomeV1):
        if record.direction is not Direction.FORWARD:
            continue
        held = latest.get(record.verified_outcome_id)
        if held is None or record.outcome_revision > held[0].outcome_revision:
            latest[record.verified_outcome_id] = (record, sequence)
    return latest


def _assessment_of(
    assessments: list[tuple[OutcomeAssessmentV1, int]], outcome: VerifiedOutcomeV1
) -> OutcomeAssessmentV1 | None:
    for record, _seq in assessments:
        if (
            record.verified_outcome_id == outcome.verified_outcome_id
            and record.outcome_revision == outcome.outcome_revision
        ):
            return record
    return None


def _feedback_of(
    feedbacks: list[tuple[DecisionFeedbackV1, int]], assessment: OutcomeAssessmentV1
) -> DecisionFeedbackV1 | None:
    for record, _seq in feedbacks:
        if record.assessment_id == assessment.assessment_id:
            return record
    return None


__all__ = [
    "MAX_FEEDBACK_SCANS_PER_STARTUP",
    "TRANSIENT_FEEDBACK_REASONS",
    "AssessmentCommitStatus",
    "FeedbackCommitStatus",
    "FeedbackOwnerV1",
    "R17FeedbackReason",
]
