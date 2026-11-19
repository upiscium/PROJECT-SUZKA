"""R19 IP-20c: the Wake claim and Wake terminal owner (IP-20 D1 r3 §2, §4-4, §4-5).

``WakeClaimOwnerV1`` owns two events and is the only writer of ``wake_claims``
and ``wake_terminals`` (through the subject.history record participant):

* ``R19_WAKE_CLAIM(wakeup_id)`` records one terminal R07 claim of a due Wake: a
  lease of at most ``r19.claim_lease_s`` and one attempt of at most three.  A
  claim is not an attempt outcome: a claim that cannot be made writes nothing
  and counts nothing;
* ``R19_WAKE_TERMINAL(wakeup_id, reason)`` closes a Wake that can never produce
  a result (``EXPIRED_WAKE`` and ``ATTEMPTS_EXHAUSTED`` are ``FAILED_TERMINAL``,
  ``SUPERSEDED`` is ``CANCELLED``).

Every check that decides whether the operation may be written runs before
PREPARE (a failure inside PREPARE is a runtime fail-stop, IP-14b §2-6): the
adopted view, the raw operations of the Wake (``wake-attempt:`` keys carry no
incarnation, so the next attempt index comes from the raw store, never from the
adopted view), the gap rule, the window, the Journal-based clock floor, the
queue wait of the claim, the live ``cycle:`` and ``wake-attempt:`` keys and the
store headroom.  Any miss is a HOLD with a closed reason and nothing written;
the Wake's own lease and expiry bound everything, so a HOLD never ends a cycle.

Only the IP-20d Scheduler submits these events; with ``r19.enabled`` false every
event HOLDs ``R19_DISABLED`` before anything is read.  Both are admitted in the
soft band (HP-8, D-R19 §8-5): a claim only starts an evaluation and a terminal
only closes a Wake, so neither creates an obligation or an external effect.
No model client is reachable from here (HP-5's "2 model calls" is 0).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Final

from suzka.r16.cycle_contracts import LogicalCycleResolutionV1
from suzka.r16.decision_issuer import _DecisionIssuer, _sequence
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import (
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
from suzka.r16.successor_policy import utc_timestamp
from suzka.r19.contracts import (
    TerminalReason,
    WakeClaimV1,
    WakeTerminalKind,
    WakeTerminalV1,
    WakeTiming,
    next_attempt_index,
)
from suzka.r19.ledger import (
    WakeEntry,
    WakeHold,
    claim_eligibility,
    cycle_key,
    derive_wake_view,
    terminal_eligibility,
)
from suzka.r19.raw_state import raw_claims_of, wake_clock_floor
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.event_journal import EventJournalInspection, TransactionKind
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionCoordinator,
)

LOGGER = logging.getLogger(__name__)

_MAX_ID: Final = 256
_SOURCES: Final = {
    AgentEventType.R19_WAKE_CLAIM: AgentEventSource.RUNTIME_R19_WAKE_CLAIM,
    AgentEventType.R19_WAKE_TERMINAL: AgentEventSource.RUNTIME_R19_WAKE_TERMINAL,
    AgentEventType.R16_WAKE_REVIEW: AgentEventSource.RUNTIME_R16_WAKE_REVIEW,
}
_TERMINAL_KIND: Final = {
    TerminalReason.EXPIRED_WAKE: WakeTerminalKind.FAILED_TERMINAL,
    TerminalReason.ATTEMPTS_EXHAUSTED: WakeTerminalKind.FAILED_TERMINAL,
    TerminalReason.SUPERSEDED: WakeTerminalKind.CANCELLED,
}


class ClaimStatus(str, Enum):
    COMMITTED = "claim_committed"
    # The cycle already has its one result: nothing is written (#373 A1-9).
    EXISTING = "claim_existing"
    HOLD = "hold"


class TerminalStatus(str, Enum):
    COMMITTED = "terminal_committed"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class _ClaimResult:
    """The ``R19_WAKE_CLAIM`` value (public-safe ids and a closed reason)."""

    status: ClaimStatus
    wakeup_id: str
    claim: WakeClaimV1 | None = None
    resolution: LogicalCycleResolutionV1 | None = None
    hold: WakeHold | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not ClaimStatus:
            raise TypeError("claim status must be exact")
        if (self.status is ClaimStatus.HOLD) != (self.hold is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if (self.status is ClaimStatus.COMMITTED) != (self.claim is not None):
            raise ValueError("exactly a committed claim carries its claim")
        if (self.status is ClaimStatus.EXISTING) != (self.resolution is not None):
            raise ValueError("exactly an existing result carries its resolution")


@dataclass(frozen=True, slots=True)
class _TerminalResult:
    """The ``R19_WAKE_TERMINAL`` value."""

    status: TerminalStatus
    wakeup_id: str
    terminal: WakeTerminalV1 | None = None
    hold: WakeHold | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not TerminalStatus:
            raise TypeError("terminal status must be exact")
        if (self.status is TerminalStatus.HOLD) != (self.hold is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if (self.status is TerminalStatus.COMMITTED) != (self.terminal is not None):
            raise ValueError("exactly a committed terminal carries its terminal")


@dataclass(frozen=True, slots=True)
class _Pending:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1
    # The review's evaluated domain state (None for a claim / terminal).
    initial_state: object | None = None


class _Hold(Exception):
    def __init__(self, reason: WakeHold) -> None:
        super().__init__(reason.value)
        self.reason = reason


class WakeCommitRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


def _valid_id(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= _MAX_ID


class _WakeOwnerBase:
    """What the claim / terminal owner and the review owner share: the nonce-bound
    submission, the classification, the PREPARE step and the pre-WAL check."""

    _EVENT_TYPES: frozenset[AgentEventType] = frozenset()

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        timing: WakeTiming,
    ) -> None:
        if (
            type(decision_issuer) is not _DecisionIssuer
            or type(runtime) is not AgentRuntime
            or decision_issuer._preview._runtime is not runtime
            or type(enabled) is not bool
            or type(timing) is not WakeTiming
        ):
            raise TypeError("unqualified Wake owner binding")
        self._decision = decision_issuer
        self._runtime = runtime
        self._store = decision_issuer._store
        self._journal = decision_issuer._journal
        self._recovery = decision_issuer._recovery
        # The live clock ``L`` of D1 §4-5 (a test or the server may replace it;
        # T is always the runtime's ``event.requested_at``).
        self._clock: Callable[[], datetime] = decision_issuer._clock
        self._enabled = enabled
        self._timing = timing
        self._inflight: tuple[object, AgentEventType, tuple[object, ...]] | None = None
        self._pending: _Pending | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def timing(self) -> WakeTiming:
        return self._timing

    # ------------------------------------------------------------ submission

    def _submit(
        self, event_type: AgentEventType, target: tuple[object, ...]
    ) -> Future[AgentEventOutcome[object]]:
        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise WakeCommitRefused("wake event already pending")
            self._inflight = (nonce, event_type, target)
            try:
                return self._commit(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(event_type, _SOURCES[event_type], handle)

    def _current_event(
        self, nonce: object
    ) -> tuple[AgentEvent, AgentEventType, tuple[object, ...]]:
        event = self._runtime.current_event()
        inflight = self._inflight
        if (
            inflight is None
            or inflight[0] is not nonce
            or event is None
            or event.event_type is not inflight[1]
            or event.event_type not in self._EVENT_TYPES
            or event.source is not _SOURCES[event.event_type]
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise WakeCommitRefused("unbound wake event")
        return event, inflight[1], inflight[2]

    def _commit(self, nonce: object) -> object:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------ shared reads

    def _classify(self) -> tuple[ReaderClassification, EventJournalInspection]:
        """One adopted classification plus the Journal evidence it was read
        with; a block or a transient read is a no-row HOLD."""

        inspection = self._journal.inspect()
        try:
            classified = SubjectHistoryReader(self._journal, self._store).classify(
                inspection=inspection
            )
        except ReaderTransient:
            raise _Hold(WakeHold.HISTORY_UNAVAILABLE) from None
        if classified.global_block is not None:
            raise _Hold(WakeHold.HISTORY_UNAVAILABLE)
        return classified, inspection

    def _raw_claims(self, incarnation: str, wakeup_id: str) -> tuple[WakeClaimV1, ...]:
        try:
            return raw_claims_of(self._store.operations(incarnation), wakeup_id)
        except (TypeError, ValueError):
            # An undecodable pending envelope: the history is not usable.
            LOGGER.warning("R19 wake owner: undecodable pending claim envelope")
            raise _Hold(WakeHold.HISTORY_UNAVAILABLE) from None

    def _live_keys(self, *keys: str) -> frozenset[str]:
        return self._store.live_keys(tuple(keys))

    # ------------------------------------------------------------ PREPARE

    def _prepare(
        self,
        event: AgentEvent,
        incarnation: str,
        operation: HistoryRecordOperationV1,
        result: Callable[[], object],
        *,
        initial_state: object | None = None,
        snapshot_hash: str | None = None,
    ) -> object:
        """The last pre-PREPARE checks (the raw-state collision pre-check and the
        headroom), then the pending state and the participant.  A collision
        that reached PREPARE would be a runtime fail-stop (IP-14b §2-6)."""

        # The store refuses PREPARE of a switched-off family (a fail-stop at
        # PREPARE): check the store's own switch, not only this owner's.
        if not self._store._r19_enabled:
            raise _Hold(WakeHold.R19_DISABLED)
        collisions = self._store.prepare_collisions(operation)
        if collisions.live_keys:
            raise _Hold(WakeHold.IN_FLIGHT)
        if collisions.decision_slot_occupied:
            raise _Hold(WakeHold.COLLISION)
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _Hold(WakeHold.HISTORY_CAPACITY) from None
        if snapshot_hash is None:
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
            initial_state,
        )
        participant = SubjectHistoryRecordParticipant(
            self._store,
            incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(result(), (participant,))

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop,
        recovered at startup (IP-9 crash matrix).  The pre-checks of the
        handlers are what keep it unreachable (D1 §6-3)."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type not in self._EVENT_TYPES:
            if pending is not None:
                raise WakeCommitRefused("stale wake event pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise WakeCommitRefused("unbound wake event pending")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise WakeCommitRefused("wake incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise WakeCommitRefused("wake snapshot changed before WAL")
        if (
            pending.initial_state is not None
            and self._decision._preview._state_snapshot() != pending.initial_state
        ):
            raise WakeCommitRefused("wake domain state changed before WAL")
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
            raise WakeCommitRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None
        self._inflight = None


class WakeClaimOwnerV1(_WakeOwnerBase):
    """The Wake claim and terminal owner (``R19_WAKE_CLAIM`` /
    ``R19_WAKE_TERMINAL``)."""

    _EVENT_TYPES = frozenset(
        {AgentEventType.R19_WAKE_CLAIM, AgentEventType.R19_WAKE_TERMINAL}
    )

    # ------------------------------------------------------------ submission

    def _submit_wake_claim(self, wakeup_id: str) -> Future[AgentEventOutcome[object]]:
        if not _valid_id(wakeup_id):
            raise ValueError("a claim names one Wake id")
        return self._submit(AgentEventType.R19_WAKE_CLAIM, (wakeup_id,))

    def _submit_terminal(
        self,
        wakeup_id: str,
        reason: TerminalReason,
        superseding_wakeup_id: str | None = None,
    ) -> Future[AgentEventOutcome[object]]:
        """``reason`` is what the Scheduler wants; the handler re-checks it.
        ``SUPERSEDED`` names the newer Wake."""

        if (
            not _valid_id(wakeup_id)
            or type(reason) is not TerminalReason
            or (reason is TerminalReason.SUPERSEDED) != (superseding_wakeup_id is not None)
            or (superseding_wakeup_id is not None and not _valid_id(superseding_wakeup_id))
        ):
            raise ValueError("a terminal names one Wake, a reason and (SUPERSEDED) its successor")
        return self._submit(
            AgentEventType.R19_WAKE_TERMINAL, (wakeup_id, reason, superseding_wakeup_id)
        )

    # ------------------------------------------------------------ handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event, event_type, target = self._current_event(nonce)
        claiming = event_type is AgentEventType.R19_WAKE_CLAIM
        wakeup_id = str(target[0])
        try:
            if claiming:
                return self._claim_checked(event, wakeup_id)
            reason, superseding = target[1], target[2]
            assert type(reason) is TerminalReason
            assert superseding is None or type(superseding) is str
            return self._terminal_checked(event, wakeup_id, reason, superseding)
        except (HistoryStoreError, AgentStateLoadError):
            self._pending = None
            return self._hold(claiming, WakeHold.HISTORY_UNAVAILABLE, wakeup_id)
        except _Hold as hold:
            self._pending = None
            return self._hold(claiming, hold.reason, wakeup_id)
        except (ValueError, TypeError):
            # A builder or decoder refused a value: nothing was prepared, and a
            # claim or terminal has no per-event cap, so it must be a logged HOLD
            # (never an event failure the Scheduler would re-submit every tick).
            self._pending = None
            LOGGER.warning("R19 wake owner: a record could not be built", exc_info=True)
            return self._hold(claiming, WakeHold.UNREPRESENTABLE, wakeup_id)

    @staticmethod
    def _hold(claiming: bool, reason: WakeHold, wakeup_id: str) -> object:
        LOGGER.info(
            "R19 %s owner result=hold reason=%s",
            "claim" if claiming else "terminal",
            reason.value,
        )
        if claiming:
            return _ClaimResult(ClaimStatus.HOLD, wakeup_id, hold=reason)
        return _TerminalResult(TerminalStatus.HOLD, wakeup_id, hold=reason)

    def _require_enabled(self) -> None:
        # Before anything is read (the IP-16..IP-19 pattern).
        if not self._enabled:
            raise _Hold(WakeHold.R19_DISABLED)

    # ------------------------------------------------------------ the claim

    def _claim_checked(self, event: AgentEvent, wakeup_id: str) -> object:
        self._require_enabled()
        # D1 §4-5: the live clock is read at the handler's first line.
        live = self._clock()
        now = event.requested_at
        sequence = _sequence(event)
        classified, inspection = self._classify()
        incarnation = classified.incarnation
        view = derive_wake_view(classified.records, utc_timestamp(now), clock_floor=None)
        entry = view.get(wakeup_id)
        raw = self._raw_claims(incarnation, wakeup_id) if entry is not None else ()
        cycle_live = entry is not None and bool(
            self._live_keys(cycle_key(incarnation, entry.wake.logical_cycle_id))
        )
        floor = wake_clock_floor(
            inspection.records,
            None if entry is None else entry.latest_claim,
            claim_lease_s=self._timing.claim_lease_s,
            before_event_id=event.event_id,
        )
        check = claim_eligibility(
            entry,
            incarnation_id=incarnation,
            now=now,
            live=live,
            floor=floor,
            raw_claims=raw,
            cycle_key_live=cycle_live,
        )
        if check.existing:
            assert entry is not None and entry.resolution is not None
            LOGGER.info("R19 claim owner result=existing")
            return _ClaimResult(
                ClaimStatus.EXISTING, wakeup_id, resolution=entry.resolution
            )
        if check.hold is not None:
            raise _Hold(check.hold)
        assert entry is not None
        index = next_attempt_index(len(raw))
        assert index is not None
        try:
            claim = WakeClaimV1.for_wake(
                entry.wake,
                index,
                claim_event_id=event.event_id,
                claim_sequence=sequence,
                claimed_utc=utc_timestamp(now),
                timing=self._timing,
            )
        except ValueError:
            # "not due or has expired": no row, not yet an attempt.
            raise _Hold(WakeHold.OUTSIDE_WAKE_WINDOW) from None
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R19_WAKE_CLAIM,
            incarnation,
            event.event_id,
            sequence,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.WAKE_CLAIM,
            claim,
        )
        LOGGER.info("R19 claim owner result=committed attempt=%d", index)
        return self._prepare(
            event,
            incarnation,
            operation,
            lambda: _ClaimResult(ClaimStatus.COMMITTED, wakeup_id, claim=claim),
        )

    # ------------------------------------------------------------ the terminal

    def _terminal_checked(
        self,
        event: AgentEvent,
        wakeup_id: str,
        reason: TerminalReason,
        superseding_id: str | None,
    ) -> object:
        self._require_enabled()
        live = self._clock()
        now = event.requested_at
        sequence = _sequence(event)
        classified, inspection = self._classify()
        incarnation = classified.incarnation
        view = derive_wake_view(classified.records, utc_timestamp(now), clock_floor=None)
        entry = view.get(wakeup_id)
        newer = None if superseding_id is None else view.get(superseding_id)
        raw = self._raw_claims(incarnation, wakeup_id) if entry is not None else ()
        cycle_live = entry is not None and bool(
            self._live_keys(cycle_key(incarnation, entry.wake.logical_cycle_id))
        )
        floor = wake_clock_floor(
            inspection.records,
            None if entry is None else entry.latest_claim,
            claim_lease_s=self._timing.claim_lease_s,
            before_event_id=event.event_id,
        )
        check = terminal_eligibility(
            entry,
            reason,
            incarnation_id=incarnation,
            now=now,
            live=live,
            floor=floor,
            raw_claims=raw,
            cycle_key_live=cycle_live,
            superseding=newer,
        )
        if check.hold is not None or check.existing:
            raise _Hold(check.hold or WakeHold.WAKE_CLOSED)
        assert entry is not None
        terminal = self._build_terminal(entry, reason, newer, event, sequence)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R19_WAKE_TERMINAL,
            incarnation,
            event.event_id,
            sequence,
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.WAKE_TERMINAL,
            terminal,
        )
        # A-4: the terminal reserves its own Wake's ``cycle:`` key (and
        # ``wake_terminals_insert`` requires exactly that).
        if (
            terminal.logical_cycle_id != entry.wake.logical_cycle_id
            or cycle_key(incarnation, entry.wake.logical_cycle_id)
            not in operation.reserved_keys
        ):
            raise _Hold(WakeHold.COLLISION)
        LOGGER.info("R19 terminal owner result=committed reason=%s", reason.value)
        return self._prepare(
            event,
            incarnation,
            operation,
            lambda: _TerminalResult(TerminalStatus.COMMITTED, wakeup_id, terminal=terminal),
        )

    @staticmethod
    def _build_terminal(
        entry: WakeEntry,
        reason: TerminalReason,
        newer: WakeEntry | None,
        event: AgentEvent,
        sequence: int,
    ) -> WakeTerminalV1:
        return WakeTerminalV1.for_wake(
            entry.wake,
            _TERMINAL_KIND[reason],
            reason,
            event_id=event.event_id,
            event_sequence=sequence,
            superseding=None if newer is None else newer.wake,
        )


__all__ = [
    "ClaimStatus",
    "TerminalStatus",
    "WakeClaimOwnerV1",
    "WakeCommitRefused",
]
