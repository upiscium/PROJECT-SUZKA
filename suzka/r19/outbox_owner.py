"""R19 IP-21b: the Outbox emitter, ``R19_WAKE_COMPLETE`` (IP-21 D1 r3 §3, §4-5).

``OutboxOwnerV1`` writes the one private Outbox message of a closed cycle
(``R19_OUTBOX_EMIT`` / ``OUTBOX_MESSAGE``) and nothing else: no Wake, claim,
resolution or ACK row.  The only submitter is the IP-20d Scheduler's
``_next_complete`` step, which derives its candidates from the closed Wakes each
tick (``inbox.emission_candidates``); there is no periodic "complete scan" event.

Handler order (D1 §4-5), the mirror of IP-20 §6, so that PREPARE never raises (a
PREPARE failure is a runtime fail-stop, IP-14b §2-6):

1. ``r19.enabled`` false -> HOLD ``R19_DISABLED`` before anything is read;
2. classify the SubjectHistory (a block or a transient read -> ``HISTORY_UNAVAILABLE``);
3. the Wake in the derived view of this event's snapshot (unknown / corrupt /
   not settled / CANCELLED);
4. **dedupe first**: the ``message_id`` derives from the dedupe inputs only, so an
   adopted message is ``NONE_LEFT`` (EXISTING, nothing written) before any time is
   read; a live raw ``outbox:<message_id>`` key is ``IN_FLIGHT``;
5. the clock: ``expires = min(T + outbox_ttl, L + 7 d)`` with ``L`` a lower bound
   from record bytes (``TOO_OLD``), ``T >= L`` and ``T >= wake_clock_floor`` (the
   Journal high-water, read here and only here: ``CLOCK_UNVERIFIABLE``), then the
   DTO builder (``UNREPRESENTABLE``) and the Context binding;
6. the mirror of every ``outbox_messages_insert`` condition, then the headroom
   and the collision pre-check, then PREPARE.

Steps 3 to 5 (and the Context binding) are ``inbox.plan_emission``.  A HOLD never
changes a Wake or a resolution: the next tick re-derives.  An exception of any
kind other than a store / state read error is a logged ``UNREPRESENTABLE`` HOLD,
never raised into the runtime.  No model client is reachable from here, and the
owner never touches the Decision issuer's ``_pending``.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Final

from suzka.r16.cycle_contracts import ResultKind
from suzka.r16.decision_issuer import _DecisionIssuer, _sequence
from suzka.r16.history_records import (
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import HistoryStoreError
from suzka.r16.successor_policy import utc_timestamp
from suzka.r19.claim_owner import WakeCommitRefused, _Hold, _valid_id, _WakeOwnerBase
from suzka.r19.contracts import (
    MAX_MESSAGE_TTL,
    OutboxMessageV1,
    OutboxTemplate,
    WakeTerminalKind,
    WakeTiming,
)
from suzka.r19.inbox import (
    OutboxHold,
    adopted_message_ids,
    plan_emission,
)
from suzka.r19.ledger import WakeEntry, WakeState, derive_wake_view
from suzka.r19.raw_state import wake_clock_floor
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.event_journal import TransactionKind

LOGGER = logging.getLogger(__name__)

# ``outbox_messages_insert``: the result kind each template may report.
_TEMPLATE_RESULT: Final = {
    OutboxTemplate.DECISION_ISSUED: ResultKind.DECISION_ISSUED,
    OutboxTemplate.DECISION_CONFIRMED: ResultKind.EXISTING_DECISION,
    OutboxTemplate.DECISION_HELD: ResultKind.HOLD,
}
# The schema's ``context_id`` CHECK (``length BETWEEN 1 AND 128``).
_MAX_CONTEXT: Final = 128


class OutboxStatus(str, Enum):
    COMMITTED = "outbox_committed"
    # The message is already adopted: nothing is written (D1 §4-5 step 4).
    EXISTING = "outbox_existing"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class _OutboxResult:
    """The ``R19_WAKE_COMPLETE`` value: a Wake id, a message id and a closed code.
    The message id is a hash of the dedupe key (no cycle, digest or text)."""

    status: OutboxStatus
    wakeup_id: str
    message_id: str | None = None
    hold: OutboxHold | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not OutboxStatus:
            raise TypeError("outbox status must be exact")
        if (self.status is OutboxStatus.HOLD) != (self.hold is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.hold is not None and type(self.hold) is not OutboxHold:
            raise TypeError("hold must be an OutboxHold")
        if self.status is not OutboxStatus.HOLD and self.message_id is None:
            # A HOLD may name the message it was about; a result always does.
            raise ValueError("a committed or existing result names its message")


class _Stop(Exception):
    """An internal no-row HOLD with a closed reason."""

    def __init__(self, reason: OutboxHold, message_id: str | None = None) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.message_id = message_id


def _hold_of(reason: str) -> OutboxHold:
    try:
        return OutboxHold(reason)
    except ValueError:
        return OutboxHold.UNREPRESENTABLE


class OutboxOwnerV1(_WakeOwnerBase):
    """The Outbox emitter (``R19_WAKE_COMPLETE``)."""

    _EVENT_TYPES = frozenset({AgentEventType.R19_WAKE_COMPLETE})

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        timing: WakeTiming,
        outbox_ttl_s: int,
    ) -> None:
        super().__init__(decision_issuer, runtime, enabled=enabled, timing=timing)
        if (
            type(outbox_ttl_s) is not int
            or isinstance(outbox_ttl_s, bool)
            or not 0 < outbox_ttl_s <= int(MAX_MESSAGE_TTL.total_seconds())
        ):
            raise ValueError("outbox_ttl_s must be an integer from 1 to 7 days")
        self._outbox_ttl_s = outbox_ttl_s

    @property
    def outbox_ttl_s(self) -> int:
        return self._outbox_ttl_s

    # ------------------------------------------------------------ submission

    def _submit_complete(self, wakeup_id: str) -> Future[AgentEventOutcome[object]]:
        if not _valid_id(wakeup_id):
            raise ValueError("a completion names one Wake id")
        return self._submit(AgentEventType.R19_WAKE_COMPLETE, (wakeup_id,))

    # ------------------------------------------------------------ handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event, _event_type, target = self._current_event(nonce)
        wakeup_id = str(target[0])
        try:
            return self._emit_checked(event, wakeup_id)
        except (HistoryStoreError, AgentStateLoadError):
            self._pending = None
            return self._hold(wakeup_id, OutboxHold.HISTORY_UNAVAILABLE, None)
        except _Stop as stop:
            self._pending = None
            return self._hold(wakeup_id, stop.reason, stop.message_id)
        except _Hold as hold:
            self._pending = None
            return self._hold(wakeup_id, _hold_of(hold.reason.value), None)
        except (ValueError, TypeError):
            # A builder or decoder refused a value: nothing was prepared. A
            # logged, permanent HOLD, never an event failure the Scheduler
            # would re-submit every tick.
            self._pending = None
            LOGGER.warning("R19 outbox owner: a record could not be built", exc_info=True)
            return self._hold(wakeup_id, OutboxHold.UNREPRESENTABLE, None)

    @staticmethod
    def _hold(wakeup_id: str, reason: OutboxHold, message_id: str | None) -> _OutboxResult:
        LOGGER.info("R19 outbox owner result=hold reason=%s", reason.value)
        return _OutboxResult(OutboxStatus.HOLD, wakeup_id, message_id, reason)

    # ------------------------------------------------------------ the emission

    def _emit_checked(self, event: AgentEvent, wakeup_id: str) -> object:
        # 1. Before anything is read (the IP-16..IP-20 pattern).
        if not self._enabled:
            raise _Stop(OutboxHold.R19_DISABLED)
        now = event.requested_at
        sequence = _sequence(event)
        # 2. One adopted classification plus the Journal evidence it was read with.
        classified, inspection = self._classify()
        incarnation = classified.incarnation
        now_utc = utc_timestamp(now)
        # 3. The Wake in the derived view of this event's snapshot.
        view = derive_wake_view(classified.records, now_utc, clock_floor=None)
        entry = view.get(wakeup_id)
        if entry is None:
            raise _Stop(OutboxHold.WAKE_UNKNOWN)
        # The Journal high-water (the only Journal read, here and not in the
        # Scheduler's pre-filter): truncated to the second, as ``T`` is.
        floor = wake_clock_floor(
            inspection.records,
            entry.latest_claim,
            claim_lease_s=self._timing.claim_lease_s,
            before_event_id=event.event_id,
        )
        # 3-5 (and the Context binding): dedupe first, then the clock, then build.
        plan = plan_emission(
            entry,
            now_utc,
            outbox_ttl_s=self._outbox_ttl_s,
            adopted_ids=adopted_message_ids(classified.records),
            clock_floor=None if floor is None else utc_timestamp(floor),
        )
        if plan.hold is OutboxHold.NONE_LEFT:
            assert plan.message_id is not None
            LOGGER.info("R19 outbox owner result=existing")
            return _OutboxResult(OutboxStatus.EXISTING, wakeup_id, plan.message_id)
        # 4. A raw ``outbox:<id>`` key that is live but not adopted (PREPARED or
        # FINALIZED, not yet terminal) is in flight; it outranks the later holds.
        if plan.message_id is not None and self._live_keys(f"outbox:{plan.message_id}"):
            raise _Stop(OutboxHold.IN_FLIGHT, plan.message_id)
        if plan.hold is not None:
            raise _Stop(plan.hold, plan.message_id)
        message = plan.message
        assert message is not None
        # 6. Every ``outbox_messages_insert`` condition, before PREPARE.
        self._mirror(entry, message, incarnation)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R19_OUTBOX_EMIT,
            incarnation,
            event.event_id,
            sequence,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.OUTBOX_MESSAGE,
            message,
        )
        LOGGER.info("R19 outbox owner result=committed template=%s", message.template.value)
        return self._prepare(
            event,
            incarnation,
            operation,
            lambda: _OutboxResult(OutboxStatus.COMMITTED, wakeup_id, message.message_id),
        )

    @staticmethod
    def _mirror(entry: WakeEntry, message: OutboxMessageV1, incarnation: str) -> None:
        """The conditions of the ``outbox_messages_insert`` trigger, checked
        against the adopted entry (a trigger that fired at FINALIZE would be a
        runtime fail-stop): the source resolution / terminal is adopted in this
        incarnation with the same digest, the template maps to its result kind
        (``CYCLE_FAILED`` only from a ``FAILED_TERMINAL``), and the Context id fits
        the schema.  A miss is ``UNREPRESENTABLE``: the same adopted input never
        passes later."""

        wake = entry.wake
        if (
            entry.state not in (WakeState.CLOSED, WakeState.FAILED)
            or message.incarnation_id != incarnation
            or wake.incarnation_id != incarnation
            or not 0 < len(message.context_id) <= _MAX_CONTEXT
            or message.context_id != wake.context_id
        ):
            raise _Stop(OutboxHold.UNREPRESENTABLE, message.message_id)
        if message.template is OutboxTemplate.CYCLE_FAILED:
            terminal = entry.terminal
            if (
                terminal is None
                or terminal.terminal is not WakeTerminalKind.FAILED_TERMINAL
                or terminal.wakeup_id != wake.wakeup_id
                or terminal.terminal_id != message.source_id
                or terminal.terminal_digest != message.result_digest
            ):
                raise _Stop(OutboxHold.UNREPRESENTABLE, message.message_id)
            return
        resolution = entry.resolution
        wanted = _TEMPLATE_RESULT.get(message.template)
        if (
            resolution is None
            or wanted is None
            or resolution.result_kind is not wanted
            or resolution.incarnation_id != incarnation
            or resolution.logical_cycle_id != message.source_id
            or resolution.result_digest != message.result_digest
        ):
            raise _Stop(OutboxHold.UNREPRESENTABLE, message.message_id)


__all__ = [
    "OutboxOwnerV1",
    "OutboxStatus",
    "WakeCommitRefused",
]
