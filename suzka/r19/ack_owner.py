"""R19 IP-21c: the Inbox ACK owner (IP-21 D1 r3 §3, §5-3, §5-4, §8, §9).

``InboxAckOwnerV1`` owns one event, ``R19_INBOX_ACK``, and is the only writer of
``outbox_acks`` (through the subject.history record participant).  The ACK is a
client-driven event submitted by the Inbox route (A-10) and **only after** the
route's read-only pre-check said a row would be written; the handler repeats the
checks under the runtime's serialization (the route's read can be stale) and
mirrors every ``outbox_acks_insert`` condition **before** PREPARE: a PREPARE or
FINALIZE raise is a runtime fail-stop (IP-14b §2-6), so every failure here is a
typed outcome, never an exception into the runtime.

What an ACK does and does not do (A-12): it writes one immutable ``outbox_acks``
row meaning "a client the SUZKA API admitted received this message".  It is not
proof that a person read it, not an approval, and not an input to cognition: no
AgentState change, no Observation, no driver nudge, and the subject is not told.
Nothing in a Decision, a Wake or a claim reads an ACK.

Read side.  ``read_inbox`` and ``check_ack`` are read-only (no event, no write,
no registry mutation) and run on the route's thread, like the R18 approval
owner's fact reads.  GET does **no** eligibility check (D1 §5-3): it matches the
``context_id`` string only.  The Context's activity and the ``client_session_id``
are checked only here, in the handler, on the runtime thread, where the
ContextRegistry is read the way the chat handler reads it (``registry.get`` only;
nothing is created, selected or ``set_current``).  If no registry is reachable
the check is skipped and the literal selector alone is recorded (A-5, HD-21-8).

With ``r19.enabled`` false the owner HOLDs ``R19_DISABLED`` before reading
anything (the IP-16..IP-20 pattern); the route is not even mounted then (A-7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import Final

from suzka.r16.history_records import (
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import HistoryStoreError
from suzka.r16.decision_issuer import _DecisionIssuer, _sequence
from suzka.r16.successor_policy import utc_timestamp
from suzka.r19.claim_owner import WakeCommitRefused, _Hold, _WakeOwnerBase
from suzka.r19.contracts import AckStatus, InboxAckV1, WakeTiming
from suzka.r19.inbox import (
    AckCheck,
    AckEligibility,
    AckHold,
    AckResult,
    InboxView,
    ack_eligibility,
    ack_selector,
    derive_inbox_view,
    is_message_id,
)
from suzka.r19.ledger import WakeHold
from suzka.r19.raw_state import wake_clock_floor
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.context import (
    ContextError,
    ContextNotFound,
    ContextRegistry,
    ContextStatus,
)
from suzka.runtime.event_journal import TransactionKind

LOGGER = logging.getLogger(__name__)

# The base owner's holds that can reach an ACK, as the ACK's closed reasons.
_HOLD_MAP: Final = {
    WakeHold.R19_DISABLED: AckHold.R19_DISABLED,
    WakeHold.HISTORY_UNAVAILABLE: AckHold.HISTORY_UNAVAILABLE,
    WakeHold.IN_FLIGHT: AckHold.IN_FLIGHT,
    WakeHold.CLOCK_UNVERIFIABLE: AckHold.CLOCK_UNVERIFIABLE,
    WakeHold.HISTORY_CAPACITY: AckHold.HISTORY_CAPACITY,
    WakeHold.COLLISION: AckHold.COLLISION,
}


@dataclass(frozen=True, slots=True)
class AckOutcome:
    """The ``R19_INBOX_ACK`` value: public-safe, no row field and no id.  The
    ACK row itself (event id, sequence, selector) is never part of it."""

    result: AckResult
    hold: AckHold | None = None

    def __post_init__(self) -> None:
        if type(self.result) is not AckResult:
            raise TypeError("result must be an AckResult")
        if (self.result is AckResult.HOLD) != (self.hold is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.hold is not None and type(self.hold) is not AckHold:
            raise TypeError("hold must be an AckHold")


@dataclass(frozen=True, slots=True)
class InboxRead:
    """A read-only answer: the Inbox of one Context, or a closed reason."""

    view: InboxView | None = None
    hold: AckHold | None = None

    def __post_init__(self) -> None:
        if (self.view is None) == (self.hold is None):
            raise ValueError("exactly one of a view and a reason")


@dataclass(frozen=True, slots=True)
class AckPrecheck:
    """The route's pre-check: the closed answer, never an event."""

    check: AckCheck | None = None
    hold: AckHold | None = None

    def __post_init__(self) -> None:
        if (self.check is None) == (self.hold is None):
            raise ValueError("exactly one of a check and a reason")


class InboxAckOwnerV1(_WakeOwnerBase):
    """The Inbox ACK owner (``R19_INBOX_ACK``)."""

    _EVENT_TYPES = frozenset({AgentEventType.R19_INBOX_ACK})

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        timing: WakeTiming,
        registry: Callable[[], ContextRegistry | None] | None = None,
    ) -> None:
        super().__init__(decision_issuer, runtime, enabled=enabled, timing=timing)
        if registry is not None and not callable(registry):
            raise TypeError("registry must be a callable")
        # ``None``: no registry is reachable; the active / session check is
        # skipped (D1 §5-3, A-5).  The callable runs on the runtime thread.
        self._registry = registry

    # ------------------------------------------------------------ read side

    def read_inbox(self, context_id: str) -> InboxRead:
        """The Inbox of exactly ``context_id`` at the live clock (D1 §5-2).
        No event, no write, no registry read, no eligibility check."""

        if not self._enabled:
            return InboxRead(hold=AckHold.R19_DISABLED)
        try:
            classified, _inspection = self._classify()
            view = derive_inbox_view(
                classified.records,
                utc_timestamp(self._clock()),
                context_id=context_id,
            )
        except _Hold as hold:
            return InboxRead(hold=_HOLD_MAP.get(hold.reason, AckHold.HISTORY_UNAVAILABLE))
        except (HistoryStoreError, AgentStateLoadError, TypeError, ValueError):
            LOGGER.warning("R19 inbox read failed", exc_info=True)
            return InboxRead(hold=AckHold.HISTORY_UNAVAILABLE)
        return InboxRead(view=view)

    def check_ack(
        self, message_id: str, context_id: str, status: AckStatus
    ) -> AckPrecheck:
        """The route's read-only pre-check (D1 §5-4 steps 3-5); never an event."""

        if not self._enabled:
            return AckPrecheck(hold=AckHold.R19_DISABLED)
        try:
            classified, inspection = self._classify()
            now = self._clock()
            now_utc = utc_timestamp(now)
            view = derive_inbox_view(classified.records, now_utc, context_id=context_id)
            live = bool(self._live_keys(f"ack:{message_id}"))
            floor = wake_clock_floor(
                inspection.records, None, claim_lease_s=self._timing.claim_lease_s
            )
            check = ack_eligibility(
                view,
                message_id,
                status,
                now_utc,
                ack_key_live=live,
                clock_floor=None if floor is None else utc_timestamp(floor),
            )
        except _Hold as hold:
            return AckPrecheck(hold=_HOLD_MAP.get(hold.reason, AckHold.HISTORY_UNAVAILABLE))
        except (HistoryStoreError, AgentStateLoadError, TypeError, ValueError):
            LOGGER.warning("R19 inbox ack pre-check failed", exc_info=True)
            return AckPrecheck(hold=AckHold.HISTORY_UNAVAILABLE)
        return AckPrecheck(check=check)

    # ------------------------------------------------------------ submission

    def ack_handler(
        self,
        message_id: str,
        context_id: str,
        status: AckStatus,
        client_session_id: str | None,
    ) -> Callable[[], object]:
        """The runtime handler for one ``R19_INBOX_ACK`` event; the route passes
        it to ``execute_with_event``.  The target is bound to a nonce, so a
        handler can run only inside its own event (as every R19 owner)."""

        if (
            not is_message_id(message_id)
            or type(context_id) is not str
            or type(status) is not AckStatus
            or (client_session_id is not None and type(client_session_id) is not str)
        ):
            raise ValueError("an ACK names one message, a Context and a status")
        nonce = object()
        target = (message_id, context_id, status, client_session_id)

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise WakeCommitRefused("ack event already pending")
            self._inflight = (nonce, AgentEventType.R19_INBOX_ACK, target)
            try:
                return self._commit(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return handle

    def _current_event(
        self, nonce: object
    ) -> tuple[AgentEvent, AgentEventType, tuple[object, ...]]:
        event = self._runtime.current_event()
        inflight = self._inflight
        if (
            inflight is None
            or inflight[0] is not nonce
            or event is None
            or event.event_type is not AgentEventType.R19_INBOX_ACK
            or event.source is not AgentEventSource.API_R19_INBOX_ACK
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise WakeCommitRefused("unbound ack event")
        return event, inflight[1], inflight[2]

    # ------------------------------------------------------------ handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event, _type, target = self._current_event(nonce)
        message_id, context_id, status, client_session_id = target
        assert type(message_id) is str and type(context_id) is str
        assert type(status) is AckStatus
        assert client_session_id is None or type(client_session_id) is str
        try:
            return self._checked(event, message_id, context_id, status, client_session_id)
        except _Hold as hold:
            self._pending = None
            return self._held(_HOLD_MAP.get(hold.reason, AckHold.HISTORY_UNAVAILABLE))
        except Exception:
            # Everything before PREPARE is a typed outcome (IP-14b §2-6): a
            # failed read, a record the builders refuse, an unreadable
            # registry.  Never an exception into the runtime for a bookkeeping
            # event, and never a blocked subject.
            self._pending = None
            LOGGER.warning("R19 ack owner: the ACK could not be checked", exc_info=True)
            return self._held(AckHold.HISTORY_UNAVAILABLE)

    @staticmethod
    def _held(reason: AckHold) -> AckOutcome:
        LOGGER.info("R19 ack owner result=hold reason=%s", reason.value)
        return AckOutcome(AckResult.HOLD, reason)

    def _checked(
        self,
        event: AgentEvent,
        message_id: str,
        context_id: str,
        status: AckStatus,
        client_session_id: str | None,
    ) -> object:
        # Before anything is read (the IP-16..IP-20 pattern).
        if not self._enabled:
            raise _Hold(WakeHold.R19_DISABLED)
        now = event.requested_at
        now_utc = utc_timestamp(now)
        sequence = _sequence(event)
        selector = ack_selector(client_session_id)
        classified, inspection = self._classify()
        incarnation = classified.incarnation
        view = derive_inbox_view(classified.records, now_utc, context_id=context_id)
        floor = wake_clock_floor(
            inspection.records,
            None,
            claim_lease_s=self._timing.claim_lease_s,
            before_event_id=event.event_id,
        )
        check = ack_eligibility(
            view,
            message_id,
            status,
            now_utc,
            ack_key_live=bool(self._live_keys(f"ack:{message_id}")),
            clock_floor=None if floor is None else utc_timestamp(floor),
        )
        if check.eligibility is AckEligibility.HOLD:
            assert check.hold is not None
            return self._held(check.hold)
        if check.eligibility is not AckEligibility.WRITE:
            return self._answer(AckResult[check.eligibility.value])
        entry = view.get(message_id)
        assert entry is not None
        # The trigger's own binding: the message of this incarnation and Context
        # (``adopted_records`` is per incarnation; checked, never assumed).
        if (
            entry.message.incarnation_id != incarnation
            or entry.message.context_id != context_id
        ):
            return self._answer(AckResult.NOT_FOUND)
        context = self._context_check(context_id, client_session_id)
        if context is not None:
            return context
        ack = InboxAckV1(
            message_id,
            context_id,
            selector,
            status,
            event.event_id,
            sequence,
            now_utc,
        )
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R19_INBOX_ACK,
            incarnation,
            event.event_id,
            sequence,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.INBOX_ACK,
            ack,
        )
        if f"ack:{message_id}" not in operation.reserved_keys:
            raise _Hold(WakeHold.COLLISION)
        LOGGER.info("R19 ack owner result=committed status=%s", status.value)
        return self._prepare(
            event,
            incarnation,
            operation,
            lambda: AckOutcome(AckResult.COMMITTED),
        )

    @staticmethod
    def _answer(result: AckResult) -> AckOutcome:
        LOGGER.info("R19 ack owner result=%s", result.value.lower())
        return AckOutcome(result)

    def _context_check(
        self, context_id: str, client_session_id: str | None
    ) -> AckOutcome | None:
        """The Context must be known and active, and a given ``client_session_id``
        must be the Context's own (as ``resolve_chat_context`` requires), only
        when a registry is reachable; read-only (``registry.get``).  An unknown
        or inactive Context is the same ``NOT_FOUND`` as an unknown message (no
        oracle); a foreign session is ``CONTEXT_CONFLICT``."""

        if self._registry is None:
            return None
        registry = self._registry()
        if registry is None:
            return None
        try:
            frame = registry.get(context_id)
        except ContextNotFound:
            return self._answer(AckResult.NOT_FOUND)
        except ContextError:
            return self._held(AckHold.HISTORY_UNAVAILABLE)
        if frame.status is not ContextStatus.ACTIVE:
            return self._answer(AckResult.NOT_FOUND)
        if client_session_id is not None and frame.source_session_id != client_session_id:
            return self._held(AckHold.CONTEXT_CONFLICT)
        return None


__all__ = [
    "AckOutcome",
    "AckPrecheck",
    "InboxAckOwnerV1",
    "InboxRead",
]
