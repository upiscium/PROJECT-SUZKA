"""R17 IP-16a-3: the forward expectation (forecast) issuer (#411 D1 r3 §1, §2, §4).

``ExpectationIssuerV1`` owns ``R17_EXPECTATION_COMMIT(intent_id | scan)`` and
is the only writer of ``action_forecasts`` (through the subject.history record
participant).  It is installed beside the R18 Intent issuer (IP-16a-2), whose
Decision issuer's store, Journal and R06 snapshot it reuses; while
``r18.enabled`` is false every event HOLDs ``R18_DISABLED`` before anything is
read, and the driver never submits one (D1 §1).

The handler runs before the event opens any R07 transaction.  It reads only
through one ``SubjectHistoryReader.classify()`` of the current incarnation
(its adopted records, the Decision chains of every lineage with a Decision
row, and the operation rows); any
failure is a HOLD with a closed reason and writes nothing:

1. the target Intent: the explicit ``intent_id``, or for ``scan`` the first
   adopted forward Intent (by processing sequence) with no adopted forecast
   (none -> ``NONE_LEFT``).  An Intent that has an operation row in this
   incarnation but is not adopted (PREPARED, or FINALIZED but not yet
   terminal) is ``NOT_YET_TERMINAL`` (transient); no row at all is
   ``INTENT_UNKNOWN``;
2. an adopted forecast for that Intent -> ``EXISTING``;
3. the Intent's Decision must be an adopted node of the Intent's lineage
   chain (terminal), and the forecast is built from the Intent and bound to
   the Decision with ``ExpectedObservationV1.check_binding`` (values read from
   the Decision and the Intent), plus the selected candidate and lineage.
   ``predicted_payload_digest`` is recomputed from the Intent's artifact bytes
   and must equal ``intent.payload_digest`` (r2 P3-12).  Any mismatch is
   ``BINDING_MISMATCH`` / ``PAYLOAD_MISMATCH``;
4. ``as_of_r07_sequence`` is the committed R06 snapshot's last processed
   sequence, as the Decision issuer computes it (< this event's sequence,
   and not before the Intent's own event);
5. the forecast keys (``forecast-intent:`` / ``forecast-key:``) reserved but
   not adopted -> ``IN_FLIGHT`` (transient); then headroom;
6. a ``CoordinatedResult`` with one ``SubjectHistoryRecordParticipant``
   carrying ``HistoryRecordOperationV1(R17_EXPECTATION_COMMIT, ACTION_FORECAST,
   ExpectedObservationV1)``.

``_validate_commit`` (after this event's own PREPARE) checks the incarnation,
the R06 snapshot hash and that SubjectHistory is exactly the evaluated rows
plus this transaction's PREPARED row, as the Intent issuer does; its failure
is a runtime fail-stop, recovered at startup (PREPARED -> tombstone,
internally committed -> roll forward).

Known limits (IP-16a-3): the ledger reader and the switching-off behaviour
are IP-16b; the Policy that refuses an Intent without a forecast is IP-17.
The result is public-safe: a status, the intent and forecast ids and a
closed reason, never request text.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Final

from suzka.r16.decision_issuer import _CONFLICT_HEADS, _DecisionIssuer, _sequence
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_codec import CommittedDecisionV1, PolicyRow
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import (
    HeadBlockReason,
    HeadStatus,
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
from suzka.r17.contracts import ExpectedObservationV1
from suzka.r18.contracts import ActionIntentV1, payload_digest
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

# The driver's startup bound on consecutive scans (D1 §2, r3 P3-6).
MAX_SCANS_PER_STARTUP: Final = 16


class ExpectationCommitStatus(str, Enum):
    COMMITTED = "forecast_committed"
    EXISTING = "forecast_existing"
    NONE_LEFT = "forecast_none_left"
    HOLD = "hold"


class R17ExpectationReason(str, Enum):
    """Why the expectation issuer held; a closed, source-free vocabulary."""

    R18_DISABLED = "r18_disabled"
    HISTORY_UNAVAILABLE = "history_unavailable"
    HISTORY_CONFLICT = "history_conflict"
    INTENT_UNKNOWN = "intent_unknown"
    # Transient: the Intent exists in this incarnation but is not adopted.
    NOT_YET_TERMINAL = "not_yet_terminal"
    DECISION_UNAVAILABLE = "decision_unavailable"
    BINDING_MISMATCH = "binding_mismatch"
    PAYLOAD_MISMATCH = "payload_mismatch"
    # Transient: a forecast key is reserved but not adopted.
    IN_FLIGHT = "in_flight"
    HISTORY_CAPACITY = "history_capacity"


# The reasons the driver retries with its bounded delays.
TRANSIENT_REASONS: Final = frozenset(
    {R17ExpectationReason.NOT_YET_TERMINAL, R17ExpectationReason.IN_FLIGHT}
)


@dataclass(frozen=True, slots=True)
class _ExpectationResult:
    """The R17_EXPECTATION_COMMIT value.

    Public-safe: a status, the intent / forecast ids and a closed reason only.
    ``forecast`` is held privately for in-process inspection.
    """

    status: ExpectationCommitStatus
    intent_id: str | None
    forecast_id: str | None
    reason: R17ExpectationReason | None
    forecast: ExpectedObservationV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not ExpectationCommitStatus:
            raise TypeError("expectation status must be exact")
        held = self.status is ExpectationCommitStatus.HOLD
        if held != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not R17ExpectationReason:
            raise TypeError("expectation reason must be an exact R17ExpectationReason")
        carries = self.status in (
            ExpectationCommitStatus.COMMITTED,
            ExpectationCommitStatus.EXISTING,
        )
        if carries != (self.forecast is not None) or carries != (self.forecast_id is not None):
            raise ValueError("only a committed or existing result carries a forecast")
        if carries and self.intent_id is None:
            raise ValueError("a forecast result names its Intent")
        if self.forecast is not None and (
            self.forecast.forecast_id != self.forecast_id
            or self.forecast.action_intent_id != self.intent_id
        ):
            raise ValueError("the result's ids are not its forecast's")

    @property
    def is_authoritative(self) -> bool:
        return self.status in (
            ExpectationCommitStatus.COMMITTED,
            ExpectationCommitStatus.EXISTING,
        )


@dataclass(frozen=True, slots=True)
class _PendingForecast:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1


class _ExpectationHold(Exception):
    def __init__(self, reason: R17ExpectationReason, intent_id: str | None = None) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.intent_id = intent_id


class _ExpectationCommitRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


class ExpectationIssuerV1:
    """The R17 forecast owner for one terminal forward ActionIntent."""

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
            raise TypeError("unqualified expectation issuer binding")
        self._decision = decision_issuer
        self._runtime = runtime
        self._store = decision_issuer._store
        self._journal = decision_issuer._journal
        self._recovery = decision_issuer._recovery
        self._enabled = enabled
        self._inflight: tuple[object, str | None] | None = None
        self._pending: _PendingForecast | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------ submission

    def _submit_expectation(
        self, intent_id: str | None
    ) -> Future[AgentEventOutcome[object]]:
        """``intent_id`` names one Intent; ``None`` is the ``scan`` form."""

        if intent_id is not None and (
            type(intent_id) is not str or not intent_id or len(intent_id) > 256
        ):
            raise ValueError("a forecast names one Intent id, or scans")
        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise _ExpectationCommitRefused("forecast commit already pending")
            self._inflight = (nonce, intent_id)
            try:
                return self._commit(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(
            AgentEventType.R17_EXPECTATION_COMMIT,
            AgentEventSource.RUNTIME_R17_EXPECTATION_COMMIT,
            handle,
        )

    def _current_event(self, nonce: object) -> AgentEvent:
        event = self._runtime.current_event()
        if (
            self._inflight is None
            or self._inflight[0] is not nonce
            or event is None
            or event.event_type is not AgentEventType.R17_EXPECTATION_COMMIT
            or event.source is not AgentEventSource.RUNTIME_R17_EXPECTATION_COMMIT
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _ExpectationCommitRefused("unbound forecast commit event")
        return event

    # ------------------------------------------------------------ handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event = self._current_event(nonce)
        assert self._inflight is not None
        intent_id = self._inflight[1]
        try:
            return self._commit_checked(event, intent_id)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(R17ExpectationReason.HISTORY_UNAVAILABLE, intent_id)
        except _ExpectationHold as hold:
            return self._hold(hold.reason, hold.intent_id or intent_id)

    @staticmethod
    def _hold(reason: R17ExpectationReason, intent_id: str | None) -> _ExpectationResult:
        LOGGER.info("R17 expectation issuer result=hold reason=%s", reason.value)
        return _ExpectationResult(ExpectationCommitStatus.HOLD, intent_id, None, reason)

    def _commit_checked(self, event: AgentEvent, intent_id: str | None) -> object:
        if not self._enabled:
            raise _ExpectationHold(R17ExpectationReason.R18_DISABLED)
        classified = self._classify()
        if intent_id is None:
            intent = self._scan_target(classified)
            if intent is None:
                LOGGER.info("R17 expectation issuer result=none_left")
                return _ExpectationResult(
                    ExpectationCommitStatus.NONE_LEFT, None, None, None
                )
        else:
            intent = self._target_intent(classified, intent_id)
        # 2. an adopted forecast for this Intent is returned as EXISTING.
        existing = _forecast_for(classified, intent.intent_id)
        if existing is not None:
            LOGGER.info("R17 expectation issuer result=existing")
            return _ExpectationResult(
                ExpectationCommitStatus.EXISTING,
                intent.intent_id,
                existing.forecast_id,
                None,
                existing,
            )
        # 3. the Intent's terminal Decision, from the same classification
        # (it classifies every lineage with a Decision row in this incarnation).
        decision = self._decision_for(classified, intent)
        snapshot_hash, as_of = self._decision._snapshot()
        if as_of < intent.admitting_sequence:
            raise _ExpectationHold(R17ExpectationReason.NOT_YET_TERMINAL, intent.intent_id)
        forecast = self._forecast(event, classified.incarnation, as_of, intent)
        _bind(forecast, decision, intent)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R17_EXPECTATION_COMMIT,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.ACTION_FORECAST,
            forecast,
        )
        # 5. keys, then headroom.
        if self._store.live_keys(operation.reserved_keys):
            raise _ExpectationHold(R17ExpectationReason.IN_FLIGHT, intent.intent_id)
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _ExpectationHold(
                R17ExpectationReason.HISTORY_CAPACITY, intent.intent_id
            ) from None
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _PendingForecast(
            event,
            classified.incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(classified.incarnation)),
            transaction_id,
            operation,
        )
        LOGGER.info("R17 expectation issuer result=committed")
        participant = SubjectHistoryRecordParticipant(
            self._store,
            classified.incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(
            _ExpectationResult(
                ExpectationCommitStatus.COMMITTED,
                intent.intent_id,
                forecast.forecast_id,
                None,
                forecast,
            ),
            (participant,),
        )

    # ------------------------------------------------------------ steps

    def _classify(self) -> ReaderClassification:
        reader = SubjectHistoryReader(self._journal, self._store)
        try:
            classified = reader.classify(inspection=self._journal.inspect())
        except ReaderTransient:
            raise _ExpectationHold(R17ExpectationReason.HISTORY_UNAVAILABLE) from None
        if classified.global_block is not None:
            raise _ExpectationHold(_blocked_reason(classified.global_block))
        return classified

    @staticmethod
    def _scan_target(classified: ReaderClassification) -> ActionIntentV1 | None:
        """The first adopted forward Intent (processing order) with no forecast."""

        forecasted = {
            record.record.action_intent_id
            for record in classified.records
            if record.record_kind is RecordKind.ACTION_FORECAST
            and type(record.record) is ExpectedObservationV1
        }
        for adopted in sorted(
            classified.records, key=lambda item: (item.processing_sequence, item.transaction_id)
        ):
            record = adopted.record
            if (
                adopted.record_kind is RecordKind.ACTION_INTENT
                and type(record) is ActionIntentV1
                and record.intent_id not in forecasted
            ):
                return record
        return None

    @staticmethod
    def _target_intent(classified: ReaderClassification, intent_id: str) -> ActionIntentV1:
        adopted = _adopted_intent(classified, intent_id)
        if adopted is not None:
            return adopted
        for row in classified.rows:
            if (
                row.operation_kind is not HistoryOperationKind.R18_INTENT_ISSUE
                or row.phase not in (HistoryPhase.PREPARED, HistoryPhase.FINALIZED)
                or row.pending_bytes is None
            ):
                continue
            try:
                operation = HistoryRecordOperationV1.from_bytes(row.pending_bytes)
            except (TypeError, ValueError):
                raise _ExpectationHold(R17ExpectationReason.HISTORY_CONFLICT, intent_id) from None
            record = operation.record
            if type(record) is ActionIntentV1 and record.intent_id == intent_id:
                raise _ExpectationHold(R17ExpectationReason.NOT_YET_TERMINAL, intent_id)
        raise _ExpectationHold(R17ExpectationReason.INTENT_UNKNOWN, intent_id)

    @staticmethod
    def _decision_for(
        classified: ReaderClassification, intent: ActionIntentV1
    ) -> CommittedDecisionV1:
        """The Intent's Decision: an adopted node of its lineage chain."""

        head = classified.heads.get(intent.lineage_scope)
        if head is None or head.status is not HeadStatus.ADOPTED:
            if head is not None and head.reason in _CONFLICT_HEADS:
                raise _ExpectationHold(R17ExpectationReason.HISTORY_CONFLICT, intent.intent_id)
            raise _ExpectationHold(R17ExpectationReason.DECISION_UNAVAILABLE, intent.intent_id)
        found = [
            node.decision
            for node in classified.chains.get(intent.lineage_scope, ())
            if node.decision.decision_id == intent.decision_id
        ]
        if len(found) != 1:
            raise _ExpectationHold(R17ExpectationReason.DECISION_UNAVAILABLE, intent.intent_id)
        return found[0]

    @staticmethod
    def _forecast(
        event: AgentEvent, incarnation: str, as_of: int, intent: ActionIntentV1
    ) -> ExpectedObservationV1:
        try:
            return ExpectedObservationV1(
                lineage_scope=intent.lineage_scope,
                incarnation_id=incarnation,
                admitting_event_id=event.event_id,
                admitting_sequence=_sequence(event),
                as_of_r07_sequence=as_of,
                decision_id=intent.decision_id,
                decision_committed_digest=intent.decision_committed_digest,
                policy_kind=intent.policy_kind,
                selected_candidate_id=intent.selected_candidate_id,
                selected_candidate_digest=intent.selected_candidate_digest,
                action_intent_id=intent.intent_id,
                action_intent_digest=intent.intent_digest,
                obligation_group_key=intent.obligation_group_key,
                forward_key=intent.forward_key,
                target_instance_id=intent.target_instance_id,
                target_namespace=intent.target_namespace,
                target_expected_revision=intent.target_expected_revision,
                predicted_payload_digest=_predicted_payload_digest(intent),
            )
        except (TypeError, ValueError):
            raise _ExpectationHold(
                R17ExpectationReason.BINDING_MISMATCH, intent.intent_id
            ) from None

    # ------------------------------------------------------------ commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type is not AgentEventType.R17_EXPECTATION_COMMIT:
            if pending is not None:
                raise _ExpectationCommitRefused("stale forecast commit pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise _ExpectationCommitRefused("unbound forecast commit pending")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _ExpectationCommitRefused("forecast incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise _ExpectationCommitRefused("forecast snapshot changed before WAL")
        rows = self._store.operations(pending.incarnation)
        own = [row for row in rows if row.transaction_id == pending.transaction_id]
        others = tuple(row for row in rows if row.transaction_id != pending.transaction_id)
        if (
            len(own) != 1
            or own[0].phase is not HistoryPhase.PREPARED
            or own[0].operation_kind is not HistoryOperationKind.R17_EXPECTATION_COMMIT
            or own[0].pending_bytes != pending.operation.canonical_bytes()
            or own[0].operation_digest != pending.operation.operation_digest
            or operations_digest(others) != pending.rows_digest
        ):
            raise _ExpectationCommitRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None


# ------------------------------------------------------------------- values


def _blocked_reason(reason: HeadBlockReason | None) -> R17ExpectationReason:
    if reason in _CONFLICT_HEADS:
        return R17ExpectationReason.HISTORY_CONFLICT
    return R17ExpectationReason.HISTORY_UNAVAILABLE


def _predicted_payload_digest(intent: ActionIntentV1) -> str:
    """The nominal prediction: the digest of the exact artifact bytes P."""

    return payload_digest(intent.artifact.payload_bytes())


def _bind(
    forecast: ExpectedObservationV1,
    decision: CommittedDecisionV1,
    intent: ActionIntentV1,
) -> None:
    """D-R17 §2 binding, with values read from the Decision and the Intent."""

    try:
        forecast.check_binding(
            decision_id=decision.decision_id,
            decision_committed_digest=decision.content_digest,
            policy_kind=decision.policy_kind,
            policy_row=decision.policy_row,
            action_intent_id=intent.intent_id,
            action_intent_digest=intent.intent_digest,
            forward_key=intent.forward_key,
            target_instance_id=intent.target_instance_id,
            target_namespace=intent.target_namespace,
            target_expected_revision=intent.target_expected_revision,
        )
    except (TypeError, ValueError):
        raise _ExpectationHold(R17ExpectationReason.BINDING_MISMATCH, intent.intent_id) from None
    selected = decision.decision.selected
    if (
        decision.lineage_scope != intent.lineage_scope
        or decision.policy_row is not PolicyRow.ROW_4
        or selected.candidate_id != forecast.selected_candidate_id
        or selected.candidate_digest != forecast.selected_candidate_digest
        or intent.policy_row is not decision.policy_row
    ):
        raise _ExpectationHold(R17ExpectationReason.BINDING_MISMATCH, intent.intent_id)
    # r2 P3-12: check_binding does not compare the payload digest.
    if forecast.predicted_payload_digest != intent.payload_digest:
        raise _ExpectationHold(R17ExpectationReason.PAYLOAD_MISMATCH, intent.intent_id)


def _adopted_intent(
    classified: ReaderClassification, intent_id: str
) -> ActionIntentV1 | None:
    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.ACTION_INTENT
            and type(record) is ActionIntentV1
            and record.intent_id == intent_id
        ):
            return record
    return None


def _forecast_for(
    classified: ReaderClassification, intent_id: str
) -> ExpectedObservationV1 | None:
    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.ACTION_FORECAST
            and type(record) is ExpectedObservationV1
            and record.action_intent_id == intent_id
        ):
            return record
    return None


__all__ = [
    "MAX_SCANS_PER_STARTUP",
    "TRANSIENT_REASONS",
    "ExpectationCommitStatus",
    "ExpectationIssuerV1",
    "R17ExpectationReason",
]
