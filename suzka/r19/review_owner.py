"""R19 IP-20c: the Wake review owner, ``R16_WAKE_REVIEW`` (IP-20 D1 r3 §5, §6).

``WakeReviewOwnerV1`` is the one event that evaluates a claimed Wake (D-R19 §4)
and the only writer of Wake-origin cycle results.  It calls the Decision issuer
outside ``PLAN_PREVIEW_REVIEW`` through the IP-20a ``_evaluate`` split, at the
event's frozen admission instant T and with the review event's own sequence as
``before_sequence`` (A-14: never the Wake's stored horizon, or a stale Wake could
not see newer evidence), and it **closes the cycle or records the attempt**:

========================  ==================  ===============================
what the review found     operation kind      cycle
========================  ==================  ===============================
a Decision                ``R16_CYCLE_DECISION``    closed (Decision + resolution
                                              in ONE operation, one serial event)
same evidence, head ok    ``R16_CYCLE_RECEIPT``     closed (EXISTING receipt)
stale / drift / deadline  ``R16_CYCLE_RECEIPT``     closed (HOLD receipt)
policy HOLD / Plan B on   ``R16_CYCLE_RECEIPT``     closed (HOLD receipt)
a self-origin Goal
a recordable refusal      ``R19_WAKE_ATTEMPT_OUTCOME``  open, counts one attempt
anything the store would  none (HOLD)         open, the attempt ends at its
reject or cannot hold                         lease
========================  ==================  ===============================

so EXISTING, DRIFT and a HOLD each end the cycle once under the same ``cycle:``
key as a Decision and a terminal; none of them can end as ``FAILED_TERMINAL``.

Order (D1 §6): the adopted view and the raw store first (a resolution that
exists is returned as EXISTING before any drift is evaluated; a terminal is a
HOLD); the no-row holds whose trigger would ABORT (``closure_admissible``, the
mirror of ``wake_attempt_outcomes_insert`` / ``logical_cycle_resolutions_insert``,
§6-4) and an attempt already over; only then the recordable refusals
(``CLOCK_UNVERIFIABLE``, ``LINEAGE_MISMATCH``, an UNAVAILABLE source, a pending
Decision slot, a moved source, ``DEFERRED_BUDGET``).  A refusal is **recorded
only if every mirror passed**: a recorded refusal for a claim that is not the
latest, or whose cycle is closed or reserved, would ABORT at PREPARE or FINALIZE
and fail-stop the runtime.  Everything is decided before the transaction opens
(IP-14b D-1, A-7); nothing is substituted inside ``prepare``.

Timing (D1 §4-5): T is ``event.requested_at``; the live clock ``L`` is read at
the first line.  The wait in the runtime queue is bounded by the claim's lease,
not by 10 s (``OUTSIDE_WAKE_WINDOW`` at ``L >= lease end``), and the Commitment
deadline is checked against ``L`` with a 15 s margin (5 s plus the handler's own
10 s budget).  The 10 s monotonic budget starts at the handler's first line and
excludes the queue wait (``DEFERRED_BUDGET``, A-13: written only here).

The review keeps its own ``_pending`` and never writes the issuer's (the
``PlanPreviewOwner`` rejects a stale preview pending, D1 §5-3).  No model client
is reachable from here.  An ActionIntent is never started from a Wake (HD-20-4
(a)): the driver's Intent stage starts only from a PLAN_PREVIEW result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import time
from typing import Final

from suzka.r16.cycle_contracts import (
    CycleClosureV1,
    ExistingDecisionReceiptV1,
    HoldReason,
    HoldReceiptV1,
    LogicalCycleResolutionV1,
    wake_review_intent_key,
)
from suzka.r16.decision_issuer import (
    _DecisionIssuer,
    _Evaluation,
    _IssuerHold,
    _sequence,
    _Unqualified,
)
from suzka.r16.history_codec import CommittedDecisionV1, policy_input_digest
from suzka.r16.history_reader import HeadStatus
from suzka.r16.history_records import (
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import (
    HistoryOperationRow,
    HistoryPhase,
    HistoryStoreError,
    decision_slot_occupied,
)
from suzka.r16.successor_policy import (
    Interpretation,
    LedgerState,
    LineageState,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorPolicyResult,
    SuccessorRow,
    as_version,
    utc_timestamp,
)
from suzka.r19.claim_owner import (
    WakeCommitRefused,
    _Hold,
    _valid_id,
    _WakeOwnerBase,
)
from suzka.r19.contracts import (
    ATTEMPT_MAX_SECONDS,
    AttemptOutcome,
    RefuseReason,
    WakeAttemptOutcomeV1,
    WakeClaimV1,
    WakeTiming,
    WakeV1,
)
from suzka.r19.ledger import (
    REVIEW_DEADLINE_MARGIN,
    ReviewStep,
    WakeEntry,
    WakeHold,
    attempt_end_key,
    closure_admissible,
    cycle_key,
    derive_wake_view,
    review_eligibility,
)
from suzka.r19.raw_state import review_started_after, wake_clock_floor
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.event_journal import TransactionKind

LOGGER = logging.getLogger(__name__)

_BUDGET_SECONDS: Final = float(ATTEMPT_MAX_SECONDS)


class ReviewStatus(str, Enum):
    DECISION_ISSUED = "review_decision_issued"
    EXISTING_DECISION = "review_existing_decision"
    HOLD_RECEIPT = "review_hold_receipt"
    # A recordable refusal or DEFERRED_BUDGET: the attempt counts, the cycle stays open.
    ATTEMPT_RECORDED = "review_attempt_recorded"
    # The cycle already had its result: nothing was written (#373 A1-9).
    EXISTING = "review_existing"
    # Nothing written; the attempt ends at its lease.
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class _ReviewResult:
    """The ``R16_WAKE_REVIEW`` value: ids and closed codes only, never source
    text, a policy input or an R15 class (#374)."""

    status: ReviewStatus
    claim_id: str
    wakeup_id: str | None = None
    hold: WakeHold | None = None
    hold_reason: HoldReason | None = None
    refuse: RefuseReason | None = None
    outcome: AttemptOutcome | None = None
    resolution: LogicalCycleResolutionV1 | None = None
    decision_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not ReviewStatus:
            raise TypeError("review status must be exact")
        if (self.status is ReviewStatus.HOLD) != (self.hold is not None):
            raise ValueError("exactly a HOLD carries a reason")
        closes = self.status in (
            ReviewStatus.DECISION_ISSUED,
            ReviewStatus.EXISTING_DECISION,
            ReviewStatus.HOLD_RECEIPT,
            ReviewStatus.EXISTING,
        )
        if closes != (self.resolution is not None):
            raise ValueError("exactly a closure or an existing result carries its resolution")
        if (self.status is ReviewStatus.ATTEMPT_RECORDED) != (self.outcome is not None):
            raise ValueError("exactly a recorded attempt carries its outcome")
        if (self.status is ReviewStatus.DECISION_ISSUED) != (self.decision_id is not None):
            raise ValueError("exactly an issued Decision carries its id")

    @property
    def closed(self) -> bool:
        return self.resolution is not None


@dataclass(slots=True)
class _Review:
    """One review's inputs; nothing here is written."""

    event: AgentEvent
    sequence: int
    live: datetime
    incarnation: str
    entry: WakeEntry
    claim: WakeClaimV1
    started: float
    snapshot_hash: str | None = None

    @property
    def wake(self) -> WakeV1:
        return self.entry.wake

    @property
    def intent_key(self) -> str:
        wake = self.wake
        return wake_review_intent_key(
            wake.logical_cycle_id,
            wake.frozen_predecessor,
            wake.policy_kind,
            wake.evidence_fingerprint,
        )


class WakeReviewOwnerV1(_WakeOwnerBase):
    """The R16 Wake review owner (``R16_WAKE_REVIEW``)."""

    _EVENT_TYPES = frozenset({AgentEventType.R16_WAKE_REVIEW})

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        timing: WakeTiming,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(decision_issuer, runtime, enabled=enabled, timing=timing)
        self._monotonic = monotonic

    # ------------------------------------------------------------ submission

    def _submit_review(self, claim_id: str) -> Future[AgentEventOutcome[object]]:
        if not _valid_id(claim_id):
            raise ValueError("a review names one claim id")
        return self._submit(AgentEventType.R16_WAKE_REVIEW, (claim_id,))

    # ------------------------------------------------------------ handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event, _event_type, target = self._current_event(nonce)
        claim_id = str(target[0])
        try:
            return self._review_checked(event, claim_id)
        except (HistoryStoreError, AgentStateLoadError):
            self._pending = None
            return self._hold(claim_id, None, WakeHold.HISTORY_UNAVAILABLE)
        except _Hold as hold:
            self._pending = None
            return self._hold(claim_id, None, hold.reason)

    @staticmethod
    def _hold(claim_id: str, wakeup_id: str | None, reason: WakeHold) -> _ReviewResult:
        LOGGER.info("R16 wake review owner result=hold reason=%s", reason.value)
        return _ReviewResult(ReviewStatus.HOLD, claim_id, wakeup_id, hold=reason)

    # ------------------------------------------------------------ the review

    def _review_checked(self, event: AgentEvent, claim_id: str) -> object:
        started = self._monotonic()
        if not self._enabled:
            raise _Hold(WakeHold.R19_DISABLED)
        live = self._clock()
        now = event.requested_at
        sequence = _sequence(event)
        classified, inspection = self._classify()
        incarnation = classified.incarnation
        view = derive_wake_view(
            classified.records, utc_timestamp(now), clock_floor=None
        )
        entry = view.by_claim_id(claim_id)
        claim = (
            None
            if entry is None
            else next((c for c in entry.claims if c.claim_id == claim_id), None)
        )
        raw = () if entry is None else self._raw_claims(incarnation, entry.wakeup_id)
        live_keys = (
            frozenset()
            if entry is None
            else self._live_keys(
                cycle_key(incarnation, entry.wake.logical_cycle_id),
                attempt_end_key(claim_id),
            )
        )
        check = review_eligibility(
            entry,
            claim_id,
            now=now,
            live=live,
            floor=wake_clock_floor(
                inspection.records,
                claim,
                claim_lease_s=self._timing.claim_lease_s,
                before_event_id=event.event_id,
            ),
            journal_lineage_id=inspection.journal_lineage_id,
            incarnation_id=incarnation,
            raw_claims=raw,
            live_keys=live_keys,
            review_started=claim is not None
            and review_started_after(
                inspection.records,
                claim.claim_sequence,
                exclude_event_id=event.event_id,
            ),
        )
        wakeup_id = None if entry is None else entry.wakeup_id
        if check.step is ReviewStep.HOLD:
            assert check.hold is not None
            raise _Hold(check.hold)
        if check.step is ReviewStep.EXISTING:
            assert entry is not None and entry.resolution is not None
            LOGGER.info("R16 wake review owner result=existing")
            return _ReviewResult(
                ReviewStatus.EXISTING,
                claim_id,
                wakeup_id,
                resolution=entry.resolution,
            )
        assert entry is not None and claim is not None
        review = _Review(event, sequence, live, incarnation, entry, claim, started)
        if check.step is ReviewStep.REFUSE:
            assert check.refuse is not None
            return self._record(review, AttemptOutcome.REFUSED, check.refuse)
        return self._evaluate_and_close(review)

    def _evaluate_and_close(self, review: _Review) -> object:
        event, wake = review.event, review.wake
        # Steps 3.. (D1 §6): the issuer's evaluation at the frozen T and the
        # review's own sequence (A-14).
        try:
            evaluated = self._decision._evaluate(
                event, wake.goal_id, now=event.requested_at, before_sequence=review.sequence
            )
        except _IssuerHold:
            # E.g. the Commitment record is gone: a source is unavailable.
            return self._record(review, AttemptOutcome.REFUSED, RefuseReason.SOURCE_UNAVAILABLE)
        except (HistoryStoreError, AgentStateLoadError):
            # A store read error is a refusal (D1 §6 pre-step 3); recording it
            # re-reads the store, and a read that fails again is a no-row HOLD.
            return self._record(review, AttemptOutcome.REFUSED, RefuseReason.SOURCE_UNAVAILABLE)
        if isinstance(evaluated, _Evaluation):
            review.snapshot_hash = evaluated.snapshot_hash
        value = evaluated.value
        if _unavailable(evaluated):
            return self._record(review, AttemptOutcome.REFUSED, RefuseReason.SOURCE_UNAVAILABLE)
        # Step 4 of the pre-steps: the 10 s budget, before the PREPARE decision.
        if self._over_budget(review):
            return self._record(review, AttemptOutcome.DEFERRED_BUDGET, None)
        # (3) the current input, in the Wake's policy version, against the
        # frozen fingerprint.  ``None`` ("cannot be expressed") is no match,
        # never an error and never a match (forward note 1).
        comparable = as_version(value, wake.policy_kind)
        if comparable is None or comparable.evidence_fingerprint() != wake.evidence_fingerprint:
            return self._close_hold(review, value, HoldReason.STALE_SOURCE)
        if not isinstance(evaluated, _Evaluation):
            return self._close_hold(review, value, HoldReason.STALE_SOURCE)
        # (4) the Commitment deadline against the live clock L with a 15 s
        # margin: the queue wait ``L - T`` is absorbed, not capped.
        try:
            self._decision._check_deadline(
                evaluated.q, now=lambda: review.live, margin=REVIEW_DEADLINE_MARGIN
            )
        except _IssuerHold:
            return self._close_hold(review, value, HoldReason.INSUFFICIENT_EVIDENCE)
        head, result = evaluated.head, evaluated.result
        # (5) a qualified head with the frozen evidence: the receipt of the
        # existing Decision, or drift (same witnesses, moved sources).
        if head is not None and head.decision is not None and evaluated.predecessor is PredecessorState.QUALIFIED:
            try:
                existing = self._decision._compare_head(head.decision, value, result)
            except _IssuerHold:
                return self._close_hold(review, value, HoldReason.DECISION_SOURCE_DRIFT)
            if existing is not None:
                return self._close_existing(review, value, head.decision, result)
        # (6) the head must be the frozen predecessor; (7) otherwise stale.
        if not _head_is_frozen(wake, evaluated):
            return self._close_hold(review, value, HoldReason.STALE_SOURCE)
        if result.is_hold:
            return self._close_hold(review, value, HoldReason.POLICY_HOLD, result.row)
        return self._issue(review, evaluated)

    # ------------------------------------------------------------ outcomes

    def _over_budget(self, review: _Review) -> bool:
        return self._monotonic() - review.started > _BUDGET_SECONDS

    def _record(
        self, review: _Review, outcome: AttemptOutcome, reason: RefuseReason | None
    ) -> object:
        """A recordable refusal or ``DEFERRED_BUDGET``: one
        ``R19_WAKE_ATTEMPT_OUTCOME``, only after every mirror of §6-4 passes."""

        record = WakeAttemptOutcomeV1(review.claim.claim_id, outcome, reason)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R19_WAKE_ATTEMPT_OUTCOME,
            review.incarnation,
            review.event.event_id,
            review.sequence,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.WAKE_ATTEMPT_OUTCOME,
            record,
        )
        LOGGER.info(
            "R16 wake review owner result=attempt_recorded outcome=%s reason=%s",
            outcome.value,
            None if reason is None else reason.value,
        )
        return self._finish(
            review,
            operation,
            lambda: _ReviewResult(
                ReviewStatus.ATTEMPT_RECORDED,
                review.claim.claim_id,
                review.wake.wakeup_id,
                refuse=reason,
                outcome=outcome,
            ),
        )

    def _close_hold(
        self,
        review: _Review,
        value: SuccessorPolicyInputV2,
        reason: HoldReason,
        row: SuccessorRow | None = None,
    ) -> object:
        try:
            receipt = HoldReceiptV1(
                review.wake.logical_cycle_id,
                review.intent_key,
                review.claim.claim_id,
                review.claim.claim_digest,
                review.claim.claim_event_id,
                review.event.event_id,
                review.sequence,
                reason,
                row,
                value.evidence_fingerprint(),
                policy_input_digest(value.canonical_bytes()),
                utc_timestamp(review.event.requested_at),
            )
            resolution = LogicalCycleResolutionV1.of_hold(review.incarnation, receipt)
            closure = CycleClosureV1(resolution, receipt)
        except (TypeError, ValueError):
            # A result the receipt cannot represent (defensive): no row.
            raise _Hold(WakeHold.UNREPRESENTABLE) from None
        LOGGER.info("R16 wake review owner result=hold_receipt reason=%s", reason.value)
        return self._close(
            review,
            closure,
            HistoryOperationKind.R16_CYCLE_RECEIPT,
            lambda: _ReviewResult(
                ReviewStatus.HOLD_RECEIPT,
                review.claim.claim_id,
                review.wake.wakeup_id,
                hold_reason=reason,
                resolution=resolution,
            ),
        )

    def _close_existing(
        self,
        review: _Review,
        value: SuccessorPolicyInputV2,
        head: CommittedDecisionV1,
        result: SuccessorPolicyResult,
    ) -> object:
        row = result.history_row
        if row is None:
            raise _Hold(WakeHold.UNREPRESENTABLE)
        try:
            receipt = ExistingDecisionReceiptV1(
                review.wake.logical_cycle_id,
                review.intent_key,
                review.claim.claim_id,
                review.claim.claim_digest,
                review.event.event_id,
                review.sequence,
                utc_timestamp(review.event.requested_at),
                # Equal to the frozen one after step (3) in the Wake's version.
                review.wake.evidence_fingerprint,
                head.decision_id,
                head.content_digest,
                row,
            )
            resolution = LogicalCycleResolutionV1.of_existing(review.incarnation, receipt)
            closure = CycleClosureV1(resolution, receipt)
        except (TypeError, ValueError):
            raise _Hold(WakeHold.UNREPRESENTABLE) from None
        LOGGER.info("R16 wake review owner result=existing_decision")
        return self._close(
            review,
            closure,
            HistoryOperationKind.R16_CYCLE_RECEIPT,
            lambda: _ReviewResult(
                ReviewStatus.EXISTING_DECISION,
                review.claim.claim_id,
                review.wake.wakeup_id,
                resolution=resolution,
                decision_id=None,
            ),
        )

    def _issue(self, review: _Review, evaluated: _Evaluation) -> object:
        """The policy returned a Decision row: build the Wake-origin Decision
        with the cycle's own ``review_intent_key`` and close with it."""

        event, wake = review.event, review.wake
        assert evaluated.incarnation is not None
        try:
            committed = self._decision._committed(
                event,
                evaluated.q,
                evaluated.value,
                evaluated.result,
                evaluated.incarnation,
                evaluated.scope,
                evaluated.head,
                review_intent_key=review.intent_key,
            )
        except _IssuerHold:
            # The issuer's own rule: Plan B is never a self-origin Goal's head.
            return self._close_hold(
                review, evaluated.value, HoldReason.PLAN_SOURCE_CONFLICT
            )
        try:
            resolution = LogicalCycleResolutionV1.of_decision(
                wake.logical_cycle_id,
                review.claim.claim_id,
                review.claim.claim_digest,
                committed,
            )
            closure = CycleClosureV1(resolution, committed)
        except (TypeError, ValueError):
            raise _Hold(WakeHold.UNREPRESENTABLE) from None
        # The slots, from raw rows (PREPARED and FINALIZED-not-terminal
        # included): a FINALIZED holder is a closed question (STALE_SOURCE), a
        # PREPARED holder is a recordable DECISION_SLOT_PENDING.
        holder = _slot_holder(self._store.operations(review.incarnation), committed)
        if holder is HistoryPhase.FINALIZED:
            return self._close_hold(review, evaluated.value, HoldReason.STALE_SOURCE)
        if holder is HistoryPhase.PREPARED:
            return self._record(
                review, AttemptOutcome.REFUSED, RefuseReason.DECISION_SLOT_PENDING
            )
        # The sources must not move while the Decision was built.
        if self._sources_moved(event, evaluated):
            return self._record(review, AttemptOutcome.REFUSED, RefuseReason.SNAPSHOT_CHANGED)
        if self._over_budget(review):
            return self._record(review, AttemptOutcome.DEFERRED_BUDGET, None)
        LOGGER.info("R16 wake review owner result=issued row=%s", evaluated.result.row.value)
        return self._close(
            review,
            closure,
            HistoryOperationKind.R16_CYCLE_DECISION,
            lambda: _ReviewResult(
                ReviewStatus.DECISION_ISSUED,
                review.claim.claim_id,
                wake.wakeup_id,
                resolution=resolution,
                decision_id=committed.decision_id,
            ),
            initial_state=evaluated.initial,
        )

    def _sources_moved(self, event: AgentEvent, evaluated: _Evaluation) -> bool:
        from suzka.r16.plan_preview import _PlanSourceConflict, _PlanSourceUnavailable

        preview = self._decision._preview
        try:
            return bool(
                preview._qualify(evaluated.q.goal_id, event) != evaluated.q
                or preview._state_snapshot() != evaluated.initial
            )
        except (_PlanSourceConflict, _PlanSourceUnavailable):
            return True

    # ------------------------------------------------------------ PREPARE

    def _close(
        self,
        review: _Review,
        closure: CycleClosureV1,
        kind: HistoryOperationKind,
        result: Callable[[], object],
        *,
        initial_state: object | None = None,
    ) -> object:
        operation = HistoryRecordOperationV1(
            kind,
            review.incarnation,
            review.event.event_id,
            review.sequence,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.CYCLE_CLOSURE,
            closure,
        )
        return self._finish(review, operation, result, initial_state=initial_state)

    def _finish(
        self,
        review: _Review,
        operation: HistoryRecordOperationV1,
        result: Callable[[], object],
        *,
        initial_state: object | None = None,
    ) -> object:
        """Every mirror of §6-4 once more over fresh raw state, then the
        collision pre-check, the headroom and PREPARE.  A mirror that fails is a
        no-row HOLD: the attempt ends at its lease."""

        wake = review.wake
        raw = self._raw_claims(review.incarnation, wake.wakeup_id)
        hold = closure_admissible(
            review.entry,
            review.claim,
            incarnation_id=review.incarnation,
            raw_claims=raw,
            live_keys=self._live_keys(
                cycle_key(review.incarnation, wake.logical_cycle_id),
                attempt_end_key(review.claim.claim_id),
            ),
        )
        if hold is not None:
            raise _Hold(hold)
        return self._prepare(
            review.event,
            review.incarnation,
            operation,
            result,
            initial_state=initial_state,
            snapshot_hash=review.snapshot_hash,
        )


def _unavailable(evaluated: _Evaluation | _Unqualified) -> bool:
    """D1 §6 pre-step 3: the only refusals.  A CONFLICT source or predecessor
    and ``LineageState.MISMATCH`` are *qualified enough to close*: their input
    never equals the Wake's frozen one, so step (3) ends the cycle as
    ``STALE_SOURCE``.  ``wake_fingerprint(...) is None`` is deliberately not the
    test: it is also None for CONFLICT states."""

    if isinstance(evaluated, _Unqualified):
        return evaluated.state is RequiredSourceState.UNAVAILABLE
    value = evaluated.value
    return (
        value.required_source is RequiredSourceState.UNAVAILABLE
        or value.predecessor is PredecessorState.UNAVAILABLE
        or value.lineage is LineageState.UNAVAILABLE
        or value.interpretation is Interpretation.UNAVAILABLE
        # HD-20-3 (a): time-varying, never frozen and never closes a cycle.
        or value.ledger is LedgerState.TARGET_UNAVAILABLE
    )


def _slot_holder(
    rows: Iterable[HistoryOperationRow], committed: CommittedDecisionV1
) -> HistoryPhase | None:
    """The phase of the raw Decision operation (PREPARED or FINALIZED, of either
    Decision kind) that holds this Decision's review-intent or predecessor slot,
    FINALIZED first; None when both slots are free (IP-14b D1 §2-6 raw pre-check)."""

    held = list(rows)
    for phase in (HistoryPhase.FINALIZED, HistoryPhase.PREPARED):
        if decision_slot_occupied(
            [row for row in held if row.phase is phase],
            lineage_scope=committed.lineage_scope,
            review_intent_key=committed.review_intent_key,
            predecessor_key=committed.predecessor_key,
        ):
            return phase
    return None


def _head_is_frozen(wake: WakeV1, evaluated: _Evaluation) -> bool:
    """(6): the head equals the Wake's frozen predecessor, a Decision by id and
    digest, or an absence equal to an empty lineage."""

    head, frozen = evaluated.head, wake.frozen_predecessor
    if head is None:
        return False
    if frozen.decision_id is None:
        return head.status is HeadStatus.EMPTY
    decision = head.decision
    return (
        head.status is HeadStatus.ADOPTED
        and decision is not None
        and decision.decision_id == frozen.decision_id
        and decision.content_digest == frozen.decision_digest
    )


__all__ = [
    "ReviewStatus",
    "WakeCommitRefused",
    "WakeReviewOwnerV1",
]
