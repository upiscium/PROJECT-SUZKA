"""R16 production driver (#370 R20-DRV D0-D2, IP-6b).

The driver only *selects and submits*. Every owner re-checks its own
sources, authority and HOLD inside its own serialized event (#369 D2-1);
the driver writes no AgentState, Journal or store, never reads a store or
an owner's live state (#370 D2 P0-1), and never overrides an owner.

Shape (#370 D0 §0):
- one dedicated thread ``r16-driver`` and a bounded FIFO (64 items);
- inputs are CHAT terminals from the chat route (``notify_chat_terminal``)
  and the done-callbacks of futures the driver itself submitted;
- callbacks only ``put_nowait`` (and count a drop on ``queue.Full``); all
  waiting, retries and submits happen on the driver thread.

Stages (#370 D0 §2, D1, D2):
- an ordinary CHAT turn whose committed appraisal is not neutral (the bit
  ``eligible_for_u2a`` from the CHAT outcome) gets one U2a submission,
  deferred while the runtime has other queued work (E: 0.2 s polls, at most
  30 s, then submitted anyway);
- ``REVIEW_EXPERIENCE:<experience id>`` and
  ``COMMIT_REVIEW_EXPERIENCE:1:<experience id>`` turns (fullmatch) get a
  ticket and are chained: proposal, review, and on adoption or acceptance
  one Plan preview per (Goal, trigger);
- a command whose target's U2a has not completed waits for it; it is
  submitted when ``U2a.sequence < P.sequence``, and otherwise ends as
  ``hold(SOURCE_NOT_YET_EVALUATED)`` so the user can resend (D2 P1-2);
- a target is resolved only from CHAT turns this incarnation observed in
  the requester's own Context; anything else is
  ``unavailable(SOURCE_NOT_OBSERVED)`` (no cross-Context existence oracle,
  D2 P2-2 / IP-5 P2-b).

Tickets (#370 D1 2-1, 2-2, D2 P1-3/P1-5, addendum P1-A): a 128-bit random
capability per command, held in a bounded in-memory ring. Nothing about
tickets is persisted or derived after a restart: a ticket from an earlier
incarnation reads ``unknown_after_restart``; an unknown ticket of this
incarnation is indistinguishable from an evicted one (404). Ticket ids are
never logged.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import queue
import re
import secrets
from threading import Event, Lock, Thread, current_thread
import time
from typing import TYPE_CHECKING

from suzka.r16.commitment_owner import _CommitmentSourceSelection
from suzka.r16.goal_owner import _GoalSourceSelection
from suzka.r16.outcomes import R16OwnerOutcome, R16OwnerReason, R16OwnerStatus
from suzka.r16.source_owner import _MotivationSourceRef
from suzka.runtime.agent_runtime import (
    AgentEventOutcome,
    AgentEventSource,
    AgentRuntime,
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeDurabilityError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
)

if TYPE_CHECKING:
    from suzka.r16.commitment_owner import _CommitmentSourceOwner
    from suzka.r16.goal_owner import _GoalSourceOwner
    from suzka.r16.plan_preview import _PlanPreviewOwner
    from suzka.r16.source_owner import _MotivationSourceOwner
    from suzka.runtime.main_loop import ChatResult

LOGGER = logging.getLogger(__name__)

QUEUE_CAPACITY = 64
OBSERVED_CAPACITY = 1024
TICKET_CAPACITY = 512
IDLE_POLL_SECONDS = 0.2
IDLE_WAIT_LIMIT_SECONDS = 30.0
RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)

_EXPERIENCE = r"experience-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_REVIEW = re.compile(rf"REVIEW_EXPERIENCE:(?P<target>{_EXPERIENCE})")
_COMMIT = re.compile(rf"COMMIT_REVIEW_EXPERIENCE:1:(?P<target>{_EXPERIENCE})")
_TICKET = re.compile(r"(?P<incarnation>[0-9a-f]{16})\.(?P<token>[A-Za-z0-9_-]{22})")


class R16CommandKind(str, Enum):
    REVIEW_EXPERIENCE = "REVIEW_EXPERIENCE"
    COMMIT_REVIEW_EXPERIENCE = "COMMIT_REVIEW_EXPERIENCE"


class R16TicketStatus(str, Enum):
    PENDING = "pending"
    ADOPTED = "adopted"
    ACCEPTED = "accepted"
    HOLD = "hold"
    UNAVAILABLE = "unavailable"
    DROPPED = "dropped"
    UNKNOWN_AFTER_RESTART = "unknown_after_restart"


class R16DriverReason(str, Enum):
    """Driver-side reasons, beside the owners' R16OwnerReason (#370 D2)."""

    SOURCE_NOT_OBSERVED = "source_not_observed"
    SOURCE_NOT_YET_EVALUATED = "source_not_yet_evaluated"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"


TicketReason = R16OwnerReason | R16DriverReason


@dataclass(frozen=True, slots=True)
class R16CommandTicket:
    """What the chat response returns for a command turn."""

    ticket_id: str
    kind: R16CommandKind
    status: R16TicketStatus = R16TicketStatus.PENDING


@dataclass(frozen=True, slots=True)
class R16TicketView:
    status: R16TicketStatus
    reason: TicketReason | None = None


@dataclass(slots=True)
class _Ticket:
    kind: R16CommandKind
    status: R16TicketStatus = R16TicketStatus.PENDING
    reason: TicketReason | None = None


@dataclass(frozen=True, slots=True)
class _Observed:
    ref: _MotivationSourceRef
    context_id: str


@dataclass(slots=True)
class _Command:
    kind: R16CommandKind
    ticket_id: str
    request: _MotivationSourceRef
    target: _MotivationSourceRef
    target_experience_id: str


@dataclass(slots=True)
class _U2a:
    experience_id: str
    ref: _MotivationSourceRef
    queued_at: float
    submitted: bool = False
    held: list[_Command] = field(default_factory=list)


class _Stage(str, Enum):
    U2A = "u2a"
    GOAL_PROPOSAL = "goal_proposal"
    GOAL_REVIEW = "goal_review"
    COMMITMENT_PROPOSAL = "commitment_proposal"
    COMMITMENT_REVIEW = "commitment_review"
    PLAN_PREVIEW = "plan_preview"


@dataclass(frozen=True, slots=True)
class _ChatTerminal:
    ref: _MotivationSourceRef
    experience_id: str | None
    context_id: str | None
    eligible_for_u2a: bool
    command: tuple[R16CommandKind, str] | None
    ticket_id: str | None


@dataclass(frozen=True, slots=True)
class _Done:
    stage: _Stage
    future: Future[AgentEventOutcome[object]]
    context: object


@dataclass(frozen=True, slots=True)
class _Retry:
    due: float
    attempt: int
    stage: _Stage
    context: object


@dataclass(frozen=True, slots=True)
class _Rescan:
    goal_ids: tuple[str, ...]
    commitment_ids: tuple[str, ...]


_STOP = object()


def parse_command(message: str) -> tuple[R16CommandKind, str] | None:
    """The command a CHAT text is, by exact fullmatch only (#370 D0 §2)."""

    if type(message) is not str:
        return None
    match = _REVIEW.fullmatch(message)
    if match is not None:
        return R16CommandKind.REVIEW_EXPERIENCE, match.group("target")
    match = _COMMIT.fullmatch(message)
    if match is not None:
        return R16CommandKind.COMMIT_REVIEW_EXPERIENCE, match.group("target")
    return None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


class R16Driver:
    """Selects and submits R16 owner events; owns no decision or authority."""

    def __init__(
        self,
        runtime: AgentRuntime,
        source_owner: _MotivationSourceOwner,
        goal_owner: _GoalSourceOwner,
        commitment_owner: _CommitmentSourceOwner,
        preview_owner: _PlanPreviewOwner,
        *,
        active: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runtime = runtime
        self._source = source_owner
        self._goal = goal_owner
        self._commitment = commitment_owner
        self._preview = preview_owner
        self._clock = clock
        self.incarnation = secrets.token_hex(8)
        self._active = active
        self._queue: queue.Queue[object] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self._stop = Event()
        self._thread: Thread | None = None
        self._lifecycle = Lock()
        self._tickets_lock = Lock()
        self._tickets: OrderedDict[str, _Ticket] = OrderedDict()
        # Driver-thread state only.
        self._observed: OrderedDict[str, _Observed] = OrderedDict()
        self._u2a: dict[str, _U2a] = {}
        self._retries: list[_Retry] = []
        self._adopted_goal: dict[str, str] = {}
        self._previewed: set[tuple[str, str]] = set()
        self._reviewed: set[str] = set()
        self._u2a_hooks: list[Callable[[_MotivationSourceRef, object], None]] = []
        self.counters: dict[str, int] = {
            "dropped": 0,
            "skipped_neutral": 0,
            "submitted": 0,
        }

    # -- lifecycle -----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Called after the unique activation event has completed."""

        self._active = True

    def start(self, rescan: _Rescan | None = None) -> None:
        with self._lifecycle:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = Thread(target=self._run, name="r16-driver", daemon=True)
            self._thread.start()
        if rescan is not None and self._active:
            self._post(rescan)

    def stop(self) -> None:
        """Stop and join before the runtime shuts down (#370 D2 P2-3)."""

        with self._lifecycle:
            thread = self._thread
            self._stop.set()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if thread is not None and thread is not current_thread():
            thread.join()
        with self._lifecycle:
            if self._thread is thread:
                self._thread = None

    def register_u2a_terminal_hook(
        self, hook: Callable[[_MotivationSourceRef, object], None]
    ) -> None:
        """#370 D0 §4 / #379: called on the driver thread after each U2a."""

        self._u2a_hooks.append(hook)

    # -- inputs ----------------------------------------------------------------

    def notify_chat_terminal(
        self, outcome: AgentEventOutcome[ChatResult], message: str
    ) -> R16CommandTicket | None:
        """Called once by the chat route right before it responds.

        Non-blocking: it only enqueues. Returns the ticket of a command turn.
        Inactive drivers, debug chat and turns without an event are no-ops.
        """

        if not self._active or self._stop.is_set():
            return None
        event = outcome.event
        result = outcome.value
        if (
            event.source is not AgentEventSource.API_CHAT
            or type(event.processing_sequence) is not int
        ):
            return None
        command = parse_command(message)
        ticket: R16CommandTicket | None = None
        if command is not None:
            ticket = self._new_ticket(command[0])
        item = _ChatTerminal(
            _MotivationSourceRef(
                event.event_id, event.processing_sequence, result.episode_id
            ),
            result.experience_id,
            result.context_id,
            result.eligible_for_u2a,
            command,
            None if ticket is None else ticket.ticket_id,
        )
        if not self._post(item) and ticket is not None:
            self._finish(ticket.ticket_id, R16TicketStatus.DROPPED)
            return R16CommandTicket(ticket.ticket_id, ticket.kind, R16TicketStatus.DROPPED)
        return ticket

    def ticket(self, ticket_id: str) -> R16TicketView | None:
        """A ticket's settled view; None for unknown or evicted (404)."""

        if type(ticket_id) is not str:
            return None
        match = _TICKET.fullmatch(ticket_id)
        if match is None:
            return None
        if match.group("incarnation") != self.incarnation:
            return R16TicketView(R16TicketStatus.UNKNOWN_AFTER_RESTART)
        with self._tickets_lock:
            found = self._tickets.get(ticket_id)
            if found is None:
                return None
            return R16TicketView(found.status, found.reason)

    # -- tickets -----------------------------------------------------------------

    def _new_ticket(self, kind: R16CommandKind) -> R16CommandTicket:
        ticket_id = f"{self.incarnation}.{secrets.token_urlsafe(16)}"
        with self._tickets_lock:
            self._tickets[ticket_id] = _Ticket(kind)
            while len(self._tickets) > TICKET_CAPACITY:
                self._tickets.popitem(last=False)
        return R16CommandTicket(ticket_id, kind)

    def _finish(
        self,
        ticket_id: str | None,
        status: R16TicketStatus,
        reason: TicketReason | None = None,
    ) -> None:
        if ticket_id is None:
            return
        with self._tickets_lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is not None and ticket.status is R16TicketStatus.PENDING:
                ticket.status = status
                ticket.reason = reason

    # -- queue -------------------------------------------------------------------

    def _post(self, item: object) -> bool:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._tickets_lock:
                self.counters["dropped"] += 1
            LOGGER.warning("R16 driver queue full; item dropped")
            return False
        return True

    def _callback(
        self, stage: _Stage, context: object
    ) -> Callable[[Future[AgentEventOutcome[object]]], None]:
        def done(future: Future[AgentEventOutcome[object]]) -> None:
            # Runs on the runtime worker (possibly under its lock) or inline:
            # enqueue only, never sleep, submit or do I/O here.
            if not self._post(_Done(stage, future, context)):
                command = context if isinstance(context, _Command) else None
                if command is not None:
                    self._finish(command.ticket_id, R16TicketStatus.DROPPED)

        return done

    # -- driver thread -------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=IDLE_POLL_SECONDS)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            try:
                if isinstance(item, _ChatTerminal):
                    self._on_chat(item)
                elif isinstance(item, _Done):
                    self._on_done(item)
                elif isinstance(item, _Rescan):
                    self._on_rescan(item)
                self._service_waiting()
            except _DriverStopped:
                break
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("R16 driver step failed")
        self._stop.set()
        self._abandon_pending()

    def _abandon_pending(self) -> None:
        with self._tickets_lock:
            for ticket in self._tickets.values():
                if ticket.status is R16TicketStatus.PENDING:
                    ticket.status = R16TicketStatus.UNAVAILABLE
                    ticket.reason = R16DriverReason.RUNTIME_UNAVAILABLE

    def _on_chat(self, item: _ChatTerminal) -> None:
        if item.experience_id is not None and item.context_id is not None:
            self._observed[item.experience_id] = _Observed(item.ref, item.context_id)
            self._observed.move_to_end(item.experience_id)
            while len(self._observed) > OBSERVED_CAPACITY:
                self._observed.popitem(last=False)
        if item.command is not None:
            self._on_command(item)
            return
        if item.experience_id is None:
            return
        if not item.eligible_for_u2a:
            self.counters["skipped_neutral"] += 1
            return
        self._u2a[item.experience_id] = _U2a(
            item.experience_id, item.ref, self._clock()
        )

    def _on_command(self, item: _ChatTerminal) -> None:
        assert item.command is not None
        kind, target_id = item.command
        observed = self._observed.get(target_id)
        if (
            observed is None
            or item.context_id is None
            or observed.context_id != item.context_id
            or observed.ref.event_sequence >= item.ref.event_sequence
        ):
            # Unobserved, or in another Context: indistinguishable.
            self._finish(
                item.ticket_id,
                R16TicketStatus.UNAVAILABLE,
                R16DriverReason.SOURCE_NOT_OBSERVED,
            )
            return
        assert item.ticket_id is not None
        command = _Command(kind, item.ticket_id, item.ref, observed.ref, target_id)
        waiting = self._u2a.get(target_id)
        if waiting is not None:
            waiting.held.append(command)
            if not waiting.submitted:
                # Stop deferring that U2a; the command still waits (D2 P1-2).
                self._submit_u2a(waiting)
            return
        self._submit_proposal(command)

    def _service_waiting(self) -> None:
        now = self._clock()
        due = [retry for retry in self._retries if retry.due <= now]
        if due:
            self._retries = [retry for retry in self._retries if retry.due > now]
            for retry in due:
                self._resubmit(retry)
        idle: bool | None = None
        for waiting in list(self._u2a.values()):
            if waiting.submitted:
                continue
            if idle is None:
                idle = self._runtime.pending_count() == 0
            if idle or now - waiting.queued_at >= IDLE_WAIT_LIMIT_SECONDS:
                self._submit_u2a(waiting)
                idle = None

    # -- submissions -------------------------------------------------------------

    def _submit(
        self, stage: _Stage, context: object, attempt: int = 0
    ) -> None:
        try:
            future = self._call(stage, context)
        except (AgentRuntimeQueueFull, AgentRuntimeAdmissionBlocked):
            if attempt < len(RETRY_DELAYS_SECONDS):
                self._retries.append(
                    _Retry(
                        self._clock() + RETRY_DELAYS_SECONDS[attempt],
                        attempt + 1,
                        stage,
                        context,
                    )
                )
            else:
                self._fail(stage, context, R16DriverReason.RUNTIME_UNAVAILABLE)
            return
        except (AgentRuntimeStopped, AgentRuntimeDurabilityError):
            self._fail(stage, context, R16DriverReason.RUNTIME_UNAVAILABLE)
            raise _DriverStopped from None
        except Exception:
            self._fail(stage, context, R16OwnerReason.SOURCE_UNAVAILABLE)
            return
        self.counters["submitted"] += 1
        LOGGER.info("R16 driver submitted stage=%s", stage.value)
        future.add_done_callback(self._callback(stage, context))

    def _resubmit(self, retry: _Retry) -> None:
        self._submit(retry.stage, retry.context, retry.attempt)

    def _call(
        self, stage: _Stage, context: object
    ) -> Future[AgentEventOutcome[object]]:
        if stage is _Stage.U2A:
            assert isinstance(context, _U2a)
            return _untyped(self._source._submit(context.ref))
        if stage is _Stage.GOAL_PROPOSAL:
            assert isinstance(context, _Command)
            return _untyped(
                self._goal._submit_proposal(
                    _GoalSourceSelection(context.request, context.target)
                )
            )
        if stage is _Stage.COMMITMENT_PROPOSAL:
            assert isinstance(context, _Command)
            return _untyped(
                self._commitment._submit_proposal(
                    _CommitmentSourceSelection(context.request, context.target)
                )
            )
        if stage is _Stage.GOAL_REVIEW:
            command, subject = _review_context(context)
            return _untyped(self._goal._submit_review(subject))
        if stage is _Stage.COMMITMENT_REVIEW:
            command, subject = _review_context(context)
            return _untyped(self._commitment._submit_review(subject))
        assert isinstance(context, str)
        return _untyped(self._preview._submit_preview(context))

    def _fail(self, stage: _Stage, context: object, reason: TicketReason) -> None:
        command = _command_of(context)
        if command is not None:
            self._finish(command.ticket_id, R16TicketStatus.UNAVAILABLE, reason)
        if stage is _Stage.U2A and isinstance(context, _U2a):
            self._u2a.pop(context.experience_id, None)
            for held in context.held:
                self._finish(held.ticket_id, R16TicketStatus.UNAVAILABLE, reason)

    def _submit_u2a(self, waiting: _U2a) -> None:
        waiting.submitted = True
        self._submit(_Stage.U2A, waiting)

    def _submit_proposal(self, command: _Command) -> None:
        self._submit(
            _Stage.GOAL_PROPOSAL
            if command.kind is R16CommandKind.REVIEW_EXPERIENCE
            else _Stage.COMMITMENT_PROPOSAL,
            command,
        )

    # -- results -------------------------------------------------------------------

    def _on_done(self, item: _Done) -> None:
        try:
            outcome = item.future.result()
        except AgentRuntimeDurabilityError:
            self._fail(item.stage, item.context, R16DriverReason.RUNTIME_UNAVAILABLE)
            raise _DriverStopped from None
        except Exception as error:
            sequence = _failed_sequence(error)
            if item.stage is _Stage.U2A and isinstance(item.context, _U2a):
                self._u2a_terminal(item.context, sequence, None)
                return
            self._fail(item.stage, item.context, R16OwnerReason.SOURCE_UNAVAILABLE)
            return
        sequence = outcome.event.processing_sequence
        if item.stage is _Stage.U2A:
            assert isinstance(item.context, _U2a)
            self._u2a_terminal(item.context, sequence, outcome.value)
            return
        if item.stage is _Stage.PLAN_PREVIEW:
            LOGGER.info("R16 driver preview completed")
            return
        value = outcome.value
        command = _command_of(item.context)
        if not isinstance(value, R16OwnerOutcome):
            self._fail(item.stage, item.context, R16OwnerReason.SOURCE_UNAVAILABLE)
            return
        ticket_id = None if command is None else command.ticket_id
        if value.status is R16OwnerStatus.HOLD:
            assert value.reason is not None
            self._finish(ticket_id, R16TicketStatus.HOLD, value.reason)
            return
        assert value.subject_id is not None
        if value.status is R16OwnerStatus.PROPOSED:
            stage = (
                _Stage.GOAL_REVIEW
                if item.stage is _Stage.GOAL_PROPOSAL
                else _Stage.COMMITMENT_REVIEW
            )
            if item.stage is _Stage.GOAL_PROPOSAL:
                self._reviewed.add(value.subject_id)
            self._submit(stage, (command, value.subject_id))
            return
        if value.status is R16OwnerStatus.ADOPTED:
            self._finish(ticket_id, R16TicketStatus.ADOPTED)
            if command is not None:
                self._adopted_goal[command.target_experience_id] = value.subject_id
            self._maybe_preview(value.subject_id, "adopted")
            return
        if value.status is R16OwnerStatus.ACCEPTED:
            self._finish(ticket_id, R16TicketStatus.ACCEPTED)
            goal_id = (
                None
                if command is None
                else self._adopted_goal.get(command.target_experience_id)
            )
            if goal_id is not None:
                self._maybe_preview(goal_id, "commitment_accepted")

    def _u2a_terminal(
        self, waiting: _U2a, sequence: int | None, value: object
    ) -> None:
        self._u2a.pop(waiting.experience_id, None)
        for command in waiting.held:
            if sequence is not None and sequence < command.request.event_sequence:
                self._submit_proposal(command)
            else:
                self._finish(
                    command.ticket_id,
                    R16TicketStatus.HOLD,
                    R16DriverReason.SOURCE_NOT_YET_EVALUATED,
                )
        for hook in self._u2a_hooks:
            try:
                hook(waiting.ref, value)
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("R16 driver U2a hook failed")

    def _maybe_preview(self, goal_id: str, trigger: str) -> None:
        key = (goal_id, trigger)
        if key in self._previewed:
            return
        self._previewed.add(key)
        self._submit(_Stage.PLAN_PREVIEW, goal_id)

    def _on_rescan(self, item: _Rescan) -> None:
        """#370 D0 §5: PROPOSED records left without a review, once each."""

        for goal_id in item.goal_ids:
            if goal_id not in self._reviewed:
                self._reviewed.add(goal_id)
                self._submit(_Stage.GOAL_REVIEW, (None, goal_id))
        for commitment_id in item.commitment_ids:
            self._submit(_Stage.COMMITMENT_REVIEW, (None, commitment_id))


class _DriverStopped(Exception):
    """The runtime stopped or failed durably; the driver stops too."""


def _untyped(future: object) -> Future[AgentEventOutcome[object]]:
    assert isinstance(future, Future)
    return future


def _review_context(context: object) -> tuple[_Command | None, str]:
    assert isinstance(context, tuple) and len(context) == 2
    command, subject = context
    assert command is None or isinstance(command, _Command)
    assert isinstance(subject, str)
    return command, subject


def _command_of(context: object) -> _Command | None:
    if isinstance(context, _Command):
        return context
    if isinstance(context, tuple) and context and isinstance(context[0], _Command):
        return context[0]
    return None


def _failed_sequence(error: BaseException) -> int | None:
    event = getattr(error, "event", None)
    sequence = getattr(event, "processing_sequence", None)
    return sequence if type(sequence) is int else None


def rescan_from(goal_state: object, commitment_state: object) -> _Rescan:
    """PROPOSED user-request Goals and PROPOSED Commitments at startup."""

    from suzka.motivation.commitment import CommitmentLifecycle
    from suzka.motivation.common import R13ReferenceKind
    from suzka.motivation.goal import GoalLifecycle

    goal_ids = tuple(
        record.goal_id
        for record in getattr(goal_state, "records", ())
        if record.lifecycle is GoalLifecycle.PROPOSED
        and any(ref.kind is R13ReferenceKind.USER_REQUEST for ref in record.origin_refs)
    )
    commitment_ids = tuple(
        record.commitment_id
        for record in getattr(commitment_state, "records", ())
        if record.lifecycle is CommitmentLifecycle.PROPOSED
    )
    return _Rescan(goal_ids, commitment_ids)


__all__ = [
    "R16CommandKind",
    "R16CommandTicket",
    "R16Driver",
    "R16DriverReason",
    "R16TicketStatus",
    "R16TicketView",
    "parse_command",
    "rescan_from",
]
