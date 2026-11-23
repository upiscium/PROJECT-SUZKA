"""R16 production driver (#370 R20-DRV D0-D2, IP-6b).

The driver only *selects and submits*. Every owner re-checks its own
sources, authority and HOLD inside its own serialized event (#369 D2-1);
the driver writes no AgentState, Journal or store, never reads a store or
an owner's live state (#370 D2 P0-1), and never overrides an owner.

Shape (#370 D0 §0):
- one dedicated thread ``r16-driver`` and a bounded FIFO (64 items);
- inputs are CHAT terminals from the chat route (``notify_chat_terminal``)
  and the done-callbacks of futures the driver itself submitted;
- callbacks only append to a lock-protected done list (never dropped: one
  entry per event the driver itself submitted) and do no I/O; all waiting,
  retries, logging and submits happen on the driver thread.

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
  D2 P2-2 / IP-5 P2-b);
- IP-16a-2 (#411 D1 §2): a Plan preview whose value is a row-4 Decision
  (ISSUED or EXISTING) gets one ``R18_ACTION_INTENT_ISSUE`` for its lineage,
  once per Decision this incarnation, only when the Intent issuer is
  installed and ``r18.enabled`` is true (the server passes no issuer
  otherwise).  An ``in_flight`` HOLD is retried with the bounded retry delays
  (at most 3 times); every other result only ends the stage;
- IP-16a-3 (#411 D1 §2): after any Intent result ISSUED or EXISTING, one
  ``R17_EXPECTATION_COMMIT(intent_id)``; a transient HOLD
  (``not_yet_terminal`` / ``in_flight``) is retried with the same bounded
  delays.  Once per driver incarnation, when the driver is (or becomes)
  active, ``R17_EXPECTATION_COMMIT(scan)``: the owner finds an adopted Intent
  without a forecast; each COMMITTED submits the scan again, and the chain
  stops at NONE_LEFT, a HOLD, or after ``MAX_SCANS_PER_STARTUP`` scans.  The
  driver never counts pending Intents itself.  Only with the expectation
  issuer installed and ``r18.enabled`` true (the server passes none
  otherwise);
- IP-17a (#411 D1 r7 §2): after any forecast result COMMITTED or EXISTING,
  one ``R18_POLICY_COMMIT(intent_id)`` per Intent this incarnation; a
  transient HOLD (``not_yet_terminal`` / ``forecast_missing`` / ``in_flight``)
  is retried with the same bounded delays (at most 3 times).  Once per
  driver incarnation, when the driver is (or becomes) active,
  ``R18_POLICY_COMMIT(scan(after))``, chained on each result's intent id (a
  HOLD on one Intent is skipped); the chain stops at NONE_LEFT, a result with
  no intent id (``r18_disabled``, a history block), a submission that is not
  admitted (the soft band), or after ``MAX_POLICY_SCANS_PER_STARTUP`` scans.
  Every HOLD is bounded anyway: an unattended Intent closes at its approval
  window's end with no event (§3a).  Only with the Policy owner installed and
  ``r18.enabled`` true;
- IP-19 (#367 D-R17 §3 / §4): once per driver incarnation when the driver is
  (or becomes) active, and again after every Outcome the executor commits
  (``notify_outcome_committed``), an ``R17_ASSESSMENT_COMMIT(scan)`` chain: each
  COMMITTED Assessment of an authorized Outcome is followed by one
  ``R17_FEEDBACK_COMMIT(assessment_id)`` (a transient HOLD is retried with the
  bounded delays) and the next Assessment scan; the chain ends at NONE_LEFT, a
  HOLD, or ``MAX_FEEDBACK_SCANS_PER_STARTUP`` links, and is followed by a
  ``R17_FEEDBACK_COMMIT(scan)`` chain that catches an Assessment whose Feedback
  was lost to a restart.  The driver never reads a store or decides eligibility.
  Only with the Feedback owner installed and ``r18.enabled`` true;
- IP-20b (IP-20 D1 r3 §3-3): an optional ``on_cause_terminal()`` callback, called
  (non-blocking, errors swallowed and logged) after a Goal is ADOPTED, a
  Commitment ACCEPTED, and an Assessment or Feedback result COMMITTED or
  EXISTING: the four causes of an R19 Wake.  The driver still imports no R19
  and reads no store; the callback is the Scheduler's nudge and is None unless
  R19 is enabled.

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
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import logging
import queue
import re
import secrets
from threading import Event, Lock, Thread, current_thread
import time
from typing import TYPE_CHECKING, Any

from suzka.r16.commands import R16CommandKind as R16CommandKind, parse_command
from suzka.r16.commitment_owner import _CommitmentSourceSelection
from suzka.r16.decision_issuer import DecisionIssueStatus, _DecisionIssueResult
from suzka.r16.goal_owner import _GoalSourceSelection, _SelfGoalSelection
from suzka.r16.outcomes import R16OwnerOutcome, R16OwnerReason, R16OwnerStatus
from suzka.r16.source_owner import INTEREST_APPLIED, _MotivationSourceRef
from suzka.r16.successor_policy import SuccessorRow
from suzka.runtime.thread_liveness import ThreadLedger
from suzka.r17.expectation_issuer import (
    MAX_SCANS_PER_STARTUP,
    TRANSIENT_REASONS,
    ExpectationCommitStatus,
    _ExpectationResult,
)
from suzka.r17.feedback_owner import (
    MAX_FEEDBACK_SCANS_PER_STARTUP,
    TRANSIENT_FEEDBACK_REASONS,
    AssessmentCommitStatus,
    FeedbackCommitStatus,
    _AssessmentResult,
    _FeedbackResult,
)
from suzka.r18.intent_issuer import (
    IntentIssueStatus,
    R18IntentReason,
    _IntentIssueResult,
)
from suzka.r18.policy_owner import (
    MAX_POLICY_SCANS_PER_STARTUP,
    TRANSIENT_POLICY_REASONS,
    PolicyCommitStatus,
    R18PolicyReason,
    _PolicyResult,
)
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
    from suzka.r17.expectation_issuer import ExpectationIssuerV1
    from suzka.r17.feedback_owner import FeedbackOwnerV1
    from suzka.r18.intent_issuer import ReviewArtifactIntentIssuerV1
    from suzka.r18.policy_owner import ReviewArtifactPolicyV1
    from suzka.r18.sweep import SweepGate
    from suzka.runtime.main_loop import ChatResult

LOGGER = logging.getLogger(__name__)

QUEUE_CAPACITY = 64
OBSERVED_CAPACITY = 1024
TICKET_CAPACITY = 512
IDLE_POLL_SECONDS = 0.2
IDLE_WAIT_LIMIT_SECONDS = 30.0
RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)
# IP-19 (condition B): a slow backstop re-runs the Assessment / Feedback chain,
# so a lost notification (full queue, spent retries) never leaves an Outcome
# without its follow-up until a restart.
FEEDBACK_BACKSTOP_SECONDS = 600.0

_TICKET = re.compile(r"(?P<incarnation>[0-9a-f]{16})\.(?P<token>[A-Za-z0-9_-]{22})")


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
    # The Context names a participant (interlocutor_key); U2a admits no
    # event-time participant identity yet, so the driver never submits it.
    PARTICIPANT_SCOPE_UNSUPPORTED = "participant_scope_unsupported"


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
    participant_scoped: bool = False


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
    # The Context of X, so a later binding finds it even after X left the
    # observed LRU (review P3-2).
    context_id: str | None = None
    submitted: bool = False
    held: list[_Command] = field(default_factory=list)


class _Stage(str, Enum):
    U2A = "u2a"
    GOAL_PROPOSAL = "goal_proposal"
    GOAL_REVIEW = "goal_review"
    COMMITMENT_PROPOSAL = "commitment_proposal"
    COMMITMENT_REVIEW = "commitment_review"
    PLAN_PREVIEW = "plan_preview"
    # IP-11 (#379): the subject's own Goal proposal after an INTEREST U2a.
    GOAL_SELF_PROPOSAL = "goal_self_proposal"
    # IP-16a-2 (#411 D1 §2): the R18 forward Intent for a row-4 Decision.
    ACTION_INTENT = "action_intent"
    # IP-16a-3 (#411 D1 §2): the R17 forecast for one Intent, or a scan.
    EXPECTATION = "expectation"
    # IP-17a (#411 D1 r7 §2): the R18 Policy verdict for one Intent, or a scan.
    POLICY = "policy"
    # IP-19 (#367 D-R17 §3 / §4): the R17 Assessment scan and Feedback.
    ASSESSMENT = "assessment"
    FEEDBACK = "feedback"


@dataclass(frozen=True, slots=True)
class _IntentTarget:
    """A row-4 Decision's public-safe identity and lineage (no store read)."""

    decision_id: str
    lineage_scope: str


@dataclass(frozen=True, slots=True)
class _ForecastTarget:
    """One Intent's id (public-safe), or ``None`` for the owner's scan."""

    intent_id: str | None


@dataclass(frozen=True, slots=True)
class _ForecastScan:
    """Queue item: start the once-per-incarnation forecast scan."""


@dataclass(frozen=True, slots=True)
class _PolicyTarget:
    """One Intent's id (public-safe), or a scan after a cursor (``scan``)."""

    intent_id: str | None
    scan: bool = False
    after: str | None = None


@dataclass(frozen=True, slots=True)
class _PolicyScan:
    """Queue item: start the once-per-incarnation Policy scan."""


@dataclass(frozen=True, slots=True)
class _AssessmentTarget:
    """The Assessment scan (the owner picks the Outcome; no ids here)."""


@dataclass(frozen=True, slots=True)
class _FeedbackTarget:
    """One Assessment's id (public-safe), or ``None`` for the owner's scan."""

    assessment_id: str | None


@dataclass(frozen=True, slots=True)
class _FeedbackScan:
    """Queue item: (re)start the Assessment -> Feedback scan chain."""


@dataclass(frozen=True, slots=True)
class _Wake:
    """Queue item: a result of the driver's own submission is ready."""


@dataclass(frozen=True, slots=True)
class _GateOpen:
    """Queue item: the AdmissionGate moved into OPEN (IP-18a, §3-6)."""


@dataclass(frozen=True, slots=True)
class _BlockedClear:
    """Queue item: BLOCKED_TARGET cleared (after its dwell; IP-18b, §3-6 item 5)."""


@dataclass(frozen=True, slots=True)
class _ChatTerminal:
    ref: _MotivationSourceRef
    experience_id: str | None
    context_id: str | None
    eligible_for_u2a: bool
    command: tuple[R16CommandKind, str] | None
    ticket_id: str | None
    participant_scoped: bool = False


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
class _SelfTarget:
    """IP-11: the committed CHAT X an INTEREST was applied to (no store read)."""

    experience_id: str
    ref: _MotivationSourceRef


@dataclass(frozen=True, slots=True)
class _Rescan:
    goal_ids: tuple[str, ...]
    commitment_ids: tuple[str, ...]
    # IP-11 (#379 D1 P1-5 / D5 P3-2): eligible INTERESTs without a Goal,
    # selected by the server before the runtime starts.
    self_targets: tuple[_SelfTarget, ...] = ()


_STOP = object()


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
        self_goal_active: bool = False,
        self_goal_per_incarnation: int = 3,
        clock: Callable[[], float] = time.monotonic,
        intent_issuer: ReviewArtifactIntentIssuerV1 | None = None,
        expectation_issuer: ExpectationIssuerV1 | None = None,
        policy_owner: ReviewArtifactPolicyV1 | None = None,
        gate: SweepGate | None = None,
        feedback_owner: FeedbackOwnerV1 | None = None,
        on_cause_terminal: Callable[[], None] | None = None,
    ) -> None:
        self._runtime = runtime
        self._source = source_owner
        self._goal = goal_owner
        self._commitment = commitment_owner
        self._preview = preview_owner
        self._clock = clock
        # IP-16a-2: present only when installed with r18.enabled (server).
        self._intent = intent_issuer
        # Decisions whose Intent was submitted, and in_flight retries made.
        self._intended: dict[str, None] = {}
        self._intent_retries: dict[str, int] = {}
        # IP-16a-3: present only when installed with r18.enabled (server).
        self._expectation = expectation_issuer
        # Intents whose forecast was submitted, transient retries made, and
        # this incarnation's scan state (D1 §2, r3 P3-6).
        self._forecasted: dict[str, None] = {}
        self._forecast_retries: dict[str, int] = {}
        self._scan_posted = False
        self._scans = 0
        # IP-17a: present only when installed with r18.enabled (server).
        self._policy = policy_owner
        # Intents whose verdict was submitted, transient retries made, and
        # this incarnation's Policy scan state (D1 r7 §2).
        self._policed: dict[str, None] = {}
        self._policy_retries: dict[str, int] = {}
        self._policy_scan_posted = False
        self._policy_scans = 0
        # IP-20b: the R19 Scheduler's nudge (None unless r19.enabled).
        self._on_cause_terminal = on_cause_terminal
        # IP-19: present only when installed with r18.enabled (server).
        self._feedback = feedback_owner
        # Assessments whose Feedback was submitted, transient retries made, and
        # this incarnation's scan state (links per chain, bounded).
        self._feedbacked: dict[str, None] = {}
        self._feedback_retries: dict[str, int] = {}
        self._scan_retries: dict[str, int] = {}
        self._next_feedback_backstop = clock() + FEEDBACK_BACKSTOP_SECONDS
        self._assess_scans = 0
        self._feedback_scans = 0
        self._feedback_scan_posted = False
        # IP-18a (#411 IP-18 D1 r4 §3-6): an Intent / Policy stage met a closed
        # AdmissionGate and was dropped from ``_intended`` / ``_policed``;
        # its target is remembered (bounded) and re-submitted on every
        # transition of the gate into OPEN.
        self._gate = gate
        self._gate_waiting: dict[str, _IntentTarget] = {}
        self._gate_subscribed = False
        # IP-18b (§3-6 item 5): (goal, trigger) previews that ran while the
        # target was blocked (bounded); re-previewed once when it clears.
        self._blocked_previews: dict[tuple[str, str], None] = {}
        self._blocked_clear_flag = Event()
        self.incarnation = secrets.token_hex(8)
        self._active = active
        # IP-11: the self path also needs its own unique activation marker.
        self._self_goal_active = self_goal_active
        self._self_goal_limit = self_goal_per_incarnation
        self._self_targets: dict[str, None] = {}
        # Admitted plus waiting-for-retry self proposals (review P2-1).
        self._self_reserved = 0
        # X a user named in a command this incarnation: the user comes first,
        # even when that command is told to resend (review P1-1).
        self._user_targets: dict[str, None] = {}
        self._queue: queue.Queue[object] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self._stop = Event()
        self._thread: Thread | None = None
        self._thread_ledger = ThreadLedger()
        self._lifecycle = Lock()
        self._tickets_lock = Lock()
        self._tickets: OrderedDict[str, _Ticket] = OrderedDict()
        # Driver-thread state only.
        self._observed: OrderedDict[str, _Observed] = OrderedDict()
        self._u2a: dict[str, _U2a] = {}
        self._retries: list[_Retry] = []
        self._adopted_goal: dict[str, str] = {}
        # Insertion-ordered, so bounding evicts the oldest (P2-b).
        self._previewed: dict[tuple[str, str], None] = {}
        self._reviewed: dict[str, None] = {}
        self._u2a_hooks: list[Callable[[_MotivationSourceRef, object], None]] = []
        # Results of the driver's own submissions (IP-6b review P1-1): never
        # dropped, so a stage can never be left without its result.
        self._done: list[_Done] = []
        self._halted = False
        self.counters: dict[str, int] = {
            "dropped": 0,
            "skipped_neutral": 0,
            "skipped_participant_scope": 0,
            "submitted": 0,
            "self_proposal_dropped": 0,
        }

    # -- lifecycle -----------------------------------------------------------

    @property
    def active(self) -> bool:
        with self._tickets_lock:
            return self._active and not self._halted

    def activate(self) -> None:
        """Called after the unique activation event has completed."""

        self._active = True
        with self._lifecycle:
            started = self._thread is not None
        if started:
            self._post_forecast_scan()
            self._post_policy_scan()
            self._post_feedback_scan()

    def activate_self_goal(self) -> None:
        """IP-11: called after the unique self-Goal activation has completed."""

        self._self_goal_active = True

    def start(self, rescan: _Rescan | None = None) -> None:
        with self._lifecycle:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = Thread(target=self._run, name="r16-driver", daemon=True)
            self._thread.start()
            self._thread_ledger.register(self._thread)
        if rescan is not None and self._active:
            self._post(rescan)
        if self._active:
            self._post_forecast_scan()
            self._post_policy_scan()
            self._post_feedback_scan()
        self._subscribe_gate()

    def _subscribe_gate(self) -> None:
        """The gate callback only posts to this driver's own queue (it runs on
        the gate's thread, possibly at once on this one); it is a no-op after
        ``stop()``."""

        if self._gate is None or self._gate_subscribed:
            return
        self._gate_subscribed = True
        self._gate.subscribe(self._on_gate_open_callback)
        subscribe_unblocked = getattr(self._gate, "subscribe_unblocked", None)
        if subscribe_unblocked is not None:
            subscribe_unblocked(self._on_blocked_clear_callback)

    def _on_blocked_clear_callback(self) -> None:
        if self._stop.is_set():
            return
        # A flag as well as a message: a full queue drops the message, and the
        # gate fires the clear only once; the idle poll sees the flag.
        self._blocked_clear_flag.set()
        self._post(_BlockedClear())

    def _on_gate_open_callback(self) -> None:
        if self._stop.is_set():
            return
        self._post(_GateOpen())

    def _post_policy_scan(self) -> None:
        """IP-17a (D1 r7 §2): one Policy scan when the driver is active,
        queued after the forecast scan."""

        if self._policy is None or not self._policy.enabled or self._policy_scan_posted:
            return
        self._policy_scan_posted = True
        self._post(_PolicyScan())

    def _post_feedback_scan(self) -> None:
        """IP-19: one Assessment / Feedback scan chain when the driver is active
        (restart recovery of an Outcome or Assessment with no follow-up)."""

        if self._feedback is None or not self._feedback.enabled or self._feedback_scan_posted:
            return
        self._feedback_scan_posted = True
        self._post(_FeedbackScan())

    def notify_outcome_committed(self) -> None:
        """IP-19: the executor committed (or found) an Outcome.  Called from the
        executor's thread: it only posts to this driver's own queue, and is a
        no-op while the driver is inactive (the activation scan then covers it)
        or stopped."""

        if (
            self._feedback is None
            or not self._feedback.enabled
            or not self._active
            or self._stop.is_set()
        ):
            return
        self._post(_FeedbackScan())

    def _cause_terminal(self) -> None:
        """IP-20b: a cause of an R19 Wake is terminal.  Non-blocking by contract
        (the Scheduler's nudge); a failing callback never stops the driver."""

        callback = self._on_cause_terminal
        if callback is None:
            return
        try:
            callback()
        except Exception:
            LOGGER.info("R16 driver cause-terminal hook failed")

    def _post_forecast_scan(self) -> None:
        """IP-16a-3 (D1 §2): one forecast scan when the driver is active."""

        if (
            self._expectation is None
            or not self._expectation.enabled
            or self._scan_posted
        ):
            return
        self._scan_posted = True
        self._post(_ForecastScan())

    def threads_alive(self) -> tuple[str, ...]:
        """Names of this owner's threads that are still alive (read-only;
        remembers a thread that ``stop()`` timed out on and abandoned)."""

        return self._thread_ledger.alive()

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
            thread.join(timeout=30)
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

        if not self.active or self._stop.is_set():
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
            if ticket is None:
                return None
        item = _ChatTerminal(
            _MotivationSourceRef(
                event.event_id, event.processing_sequence, result.episode_id
            ),
            result.experience_id,
            result.context_id,
            result.eligible_for_u2a,
            command,
            None if ticket is None else ticket.ticket_id,
            result.participant_scoped,
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

    def _new_ticket(self, kind: R16CommandKind) -> R16CommandTicket | None:
        ticket_id = f"{self.incarnation}.{secrets.token_urlsafe(16)}"
        with self._tickets_lock:
            if self._halted:
                # The driver stopped (review P2-3): no ticket can stay pending.
                return None
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
        """Enqueue a CHAT terminal or rescan from a request/lifespan thread."""
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
            # one append under a short lock; no sleep, submit or I/O here.
            with self._tickets_lock:
                self._done.append(_Done(stage, future, context))
            # IP-18a review P2-1: wake the loop (non-blocking, never counted as
            # a drop), so a result is not seen only at the next 0.2 s poll.
            try:
                self._queue.put_nowait(_Wake())
            except queue.Full:
                pass

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
                elif isinstance(item, _Rescan):
                    self._on_rescan(item)
                elif isinstance(item, _ForecastScan):
                    self._maybe_scan()
                elif isinstance(item, _PolicyScan):
                    self._maybe_policy_scan(None)
                elif isinstance(item, _FeedbackScan):
                    self._on_feedback_scan()
                elif isinstance(item, _GateOpen):
                    self._on_gate_open()
                elif isinstance(item, _BlockedClear):
                    self._on_blocked_clear()
                if self._blocked_clear_flag.is_set():
                    self._on_blocked_clear()
                self._feedback_backstop()
                with self._tickets_lock:
                    finished, self._done = self._done, []
                for result in finished:
                    # One failing result must not lose the rest (P2-a).
                    try:
                        self._on_done(result)
                    except _DriverStopped:
                        raise
                    except Exception:  # pragma: no cover - defensive
                        LOGGER.exception("R16 driver result handling failed")
                self._service_waiting()
            except _DriverStopped:
                break
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("R16 driver step failed")
        self._stop.set()
        self._abandon_pending()

    def _abandon_pending(self) -> None:
        with self._tickets_lock:
            self._halted = True
            for ticket in self._tickets.values():
                if ticket.status is R16TicketStatus.PENDING:
                    ticket.status = R16TicketStatus.UNAVAILABLE
                    ticket.reason = R16DriverReason.RUNTIME_UNAVAILABLE

    def _on_chat(self, item: _ChatTerminal) -> None:
        if item.experience_id is not None and item.context_id is not None:
            self._observed[item.experience_id] = _Observed(
                item.ref, item.context_id, item.participant_scoped
            )
            self._observed.move_to_end(item.experience_id)
            while len(self._observed) > OBSERVED_CAPACITY:
                self._observed.popitem(last=False)
        if item.participant_scoped and item.context_id is not None:
            self._scope_context(item.context_id)
        if item.command is not None:
            self._on_command(item)
            return
        if item.experience_id is None:
            return
        if item.participant_scoped:
            self.counters["skipped_participant_scope"] += 1
            return
        if not item.eligible_for_u2a:
            self.counters["skipped_neutral"] += 1
            return
        self._u2a[item.experience_id] = _U2a(
            item.experience_id, item.ref, self._clock(), item.context_id
        )

    def _scope_context(self, context_id: str) -> None:
        """A participant is now bound to this Context (review P2-1).

        Participant refs are never removed, so every earlier turn of the
        Context is scoped from here on: a U2a still waiting there would fail
        in the owner, so it is dropped. A U2a already submitted may still
        fail; that window is accepted. A command on a waiting U2a submits it
        at once (D2 P1-2), so a dropped U2a holds no commands today; the HOLD
        loop below only guards that invariant.
        """
        for experience_id, observed in list(self._observed.items()):
            if observed.context_id == context_id and not observed.participant_scoped:
                # Same key: the LRU order is kept.
                self._observed[experience_id] = replace(
                    observed, participant_scoped=True
                )
        for experience_id, waiting in list(self._u2a.items()):
            if waiting.submitted or waiting.context_id != context_id:
                continue
            del self._u2a[experience_id]
            self.counters["skipped_participant_scope"] += 1
            for command in waiting.held:
                self._finish(
                    command.ticket_id,
                    R16TicketStatus.HOLD,
                    R16DriverReason.PARTICIPANT_SCOPE_UNSUPPORTED,
                )

    def _on_command(self, item: _ChatTerminal) -> None:
        assert item.command is not None
        kind, target_id = item.command
        self._user_targets[target_id] = None
        _bound(self._user_targets)
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
        if observed.participant_scoped or item.participant_scoped:
            # Same Context as the requester (checked above): no oracle.
            self._finish(
                item.ticket_id,
                R16TicketStatus.HOLD,
                R16DriverReason.PARTICIPANT_SCOPE_UNSUPPORTED,
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
                self._release_unadmitted(stage)
                self._fail(stage, context, R16DriverReason.RUNTIME_UNAVAILABLE)
            return
        except (AgentRuntimeStopped, AgentRuntimeDurabilityError):
            self._fail(stage, context, R16DriverReason.RUNTIME_UNAVAILABLE)
            raise _DriverStopped from None
        except Exception:
            self._release_unadmitted(stage)
            self._fail(stage, context, R16OwnerReason.SOURCE_UNAVAILABLE)
            return
        self.counters["submitted"] += 1
        LOGGER.info("R16 driver submitted stage=%s", stage.value)
        future.add_done_callback(self._callback(stage, context))

    def _release_unadmitted(self, stage: _Stage) -> None:
        """A self proposal that was never admitted frees its reservation;
        an admitted one keeps counting even if its event fails (D3 P2-1)."""

        if stage is _Stage.GOAL_SELF_PROPOSAL:
            self._self_reserved = max(0, self._self_reserved - 1)

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
        if stage is _Stage.GOAL_SELF_PROPOSAL:
            assert isinstance(context, _SelfTarget)
            return _untyped(
                self._goal._submit_self_proposal(_SelfGoalSelection(context.ref))
            )
        if stage is _Stage.GOAL_REVIEW:
            command, subject = _review_context(context)
            return _untyped(self._goal._submit_review(subject))
        if stage is _Stage.COMMITMENT_REVIEW:
            command, subject = _review_context(context)
            return _untyped(self._commitment._submit_review(subject))
        if stage is _Stage.ACTION_INTENT:
            assert isinstance(context, _IntentTarget) and self._intent is not None
            return _untyped(self._intent._submit_intent(context.lineage_scope))
        if stage is _Stage.EXPECTATION:
            assert isinstance(context, _ForecastTarget) and self._expectation is not None
            return _untyped(self._expectation._submit_expectation(context.intent_id))
        if stage is _Stage.POLICY:
            assert isinstance(context, _PolicyTarget) and self._policy is not None
            if context.scan:
                return _untyped(self._policy._submit_scan(context.after))
            assert context.intent_id is not None
            return _untyped(self._policy._submit_policy(context.intent_id))
        if stage is _Stage.ASSESSMENT:
            assert isinstance(context, _AssessmentTarget) and self._feedback is not None
            return _untyped(self._feedback._submit_assessment(None))
        if stage is _Stage.FEEDBACK:
            assert isinstance(context, _FeedbackTarget) and self._feedback is not None
            return _untyped(self._feedback._submit_feedback(context.assessment_id))
        assert isinstance(context, str)
        return _untyped(self._preview._submit_preview(context))

    def _fail(self, stage: _Stage, context: object, reason: TicketReason) -> None:
        if stage is _Stage.GOAL_SELF_PROPOSAL:
            self.counters["self_proposal_dropped"] += 1
            LOGGER.warning("R16 driver self proposal dropped")
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
        except (AgentRuntimeDurabilityError, AgentRuntimeStopped):
            # The runtime is gone or fail-stopped (review P2-2): stop too.
            self._fail(item.stage, item.context, R16DriverReason.RUNTIME_UNAVAILABLE)
            raise _DriverStopped from None
        except Exception as error:
            LOGGER.info("R16 driver stage=%s result=error", item.stage.value)
            sequence = _failed_sequence(error)
            if item.stage is _Stage.U2A and isinstance(item.context, _U2a):
                self._u2a_terminal(item.context, sequence, None)
                return
            self._fail(item.stage, item.context, R16OwnerReason.SOURCE_UNAVAILABLE)
            return
        sequence = outcome.event.processing_sequence
        LOGGER.info(
            "R16 driver stage=%s ref=%s result=%s",
            item.stage.value,
            _digest(outcome.event.event_id),
            _result_code(outcome.value),
        )
        if item.stage is _Stage.U2A:
            assert isinstance(item.context, _U2a)
            self._u2a_terminal(item.context, sequence, outcome.value)
            return
        if item.stage is _Stage.PLAN_PREVIEW:
            LOGGER.info("R16 driver preview completed")
            self._note_blocked_result(item.context, outcome.value)
            self._maybe_intent(outcome.value)
            return
        if item.stage is _Stage.ACTION_INTENT:
            assert isinstance(item.context, _IntentTarget)
            self._intent_done(item.context, outcome.value)
            return
        if item.stage is _Stage.EXPECTATION:
            assert isinstance(item.context, _ForecastTarget)
            self._forecast_done(item.context, outcome.value)
            return
        if item.stage is _Stage.POLICY:
            assert isinstance(item.context, _PolicyTarget)
            self._policy_done(item.context, outcome.value)
            return
        if item.stage is _Stage.ASSESSMENT:
            assert isinstance(item.context, _AssessmentTarget)
            self._assessment_done(outcome.value)
            return
        if item.stage is _Stage.FEEDBACK:
            assert isinstance(item.context, _FeedbackTarget)
            self._feedback_done(item.context, outcome.value)
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
            goal_stage = item.stage in (_Stage.GOAL_PROPOSAL, _Stage.GOAL_SELF_PROPOSAL)
            stage = _Stage.GOAL_REVIEW if goal_stage else _Stage.COMMITMENT_REVIEW
            if goal_stage:
                self._reviewed[value.subject_id] = None
                _bound(self._reviewed)
            self._submit(stage, (command, value.subject_id))
            return
        if value.status is R16OwnerStatus.ADOPTED:
            self._finish(ticket_id, R16TicketStatus.ADOPTED)
            if command is not None:
                self._adopted_goal[command.target_experience_id] = value.subject_id
                while len(self._adopted_goal) > OBSERVED_CAPACITY:
                    self._adopted_goal.pop(next(iter(self._adopted_goal)))
            self._maybe_preview(value.subject_id, "adopted")
            self._cause_terminal()
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
            self._cause_terminal()

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
        # IP-11: user commands on X were submitted first (above); a user who
        # named X this incarnation (even one told to resend) keeps X.
        if value == INTEREST_APPLIED and not waiting.held:
            self._maybe_self_propose(_SelfTarget(waiting.experience_id, waiting.ref))
        for hook in self._u2a_hooks:
            try:
                hook(waiting.ref, value)
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("R16 driver U2a hook failed")

    def _maybe_self_propose(self, target: _SelfTarget) -> None:
        """#379 D2 P0: once per X and ``per_incarnation`` in total."""

        if (
            not self._self_goal_active
            or target.experience_id in self._self_targets
            or target.experience_id in self._user_targets
            or self._self_reserved >= self._self_goal_limit
        ):
            return
        self._self_targets[target.experience_id] = None
        _bound(self._self_targets)
        # Reserved before submission, so retries can never exceed the cap
        # (#379 D2 P0 / D3 P2-1); a final drop releases it in ``_fail``.
        self._self_reserved += 1
        self._submit(_Stage.GOAL_SELF_PROPOSAL, target)

    def _maybe_preview(self, goal_id: str, trigger: str) -> None:
        key = (goal_id, trigger)
        if key in self._previewed:
            return
        self._previewed[key] = None
        _bound(self._previewed)
        blocked = getattr(self._gate, "blocked_target", None)
        if blocked is not None and blocked():
            # Sealed as Plan A (TARGET_UNAVAILABLE) at best: remember it, so the
            # clear can bring the subject back to row 4 without an unrelated
            # trigger (N4).
            self._blocked_previews[key] = None
            _bound(self._blocked_previews)
        self._submit(_Stage.PLAN_PREVIEW, goal_id)

    def _note_blocked_result(self, goal_id: object, value: object) -> None:
        """P3-2: the block can begin after the preview was submitted but before
        the issuer read it. A Plan A (row 2c) result seen while the witness is
        latched is remembered too, so the clear re-previews that goal."""

        blocked = getattr(self._gate, "blocked_target", None)
        if (
            blocked is None
            or not isinstance(goal_id, str)
            or getattr(value, "row", None) is not SuccessorRow.ROW_2C
            or not blocked()
            or any(goal == goal_id for goal, _trigger in self._blocked_previews)
        ):
            return
        self._blocked_previews[(goal_id, "blocked_result")] = None
        _bound(self._blocked_previews)

    def _on_blocked_clear(self) -> None:
        """§3-6 item 5: BLOCKED_TARGET cleared (after the dwell): clear the
        ``_previewed`` keys of the goals previewed while blocked and post ONE
        PLAN_PREVIEW per such goal. One round per clear, and a clear needs
        5 continuous minutes of OPEN, so at most one return per dwell."""

        self._blocked_clear_flag.clear()
        pairs = list(self._blocked_previews)
        self._blocked_previews.clear()
        first: dict[str, str] = {}
        for goal_id, trigger in pairs:
            self._previewed.pop((goal_id, trigger), None)
            first.setdefault(goal_id, trigger)
        for goal_id, trigger in first.items():
            self._maybe_preview(goal_id, trigger)

    def _maybe_intent(self, value: object) -> None:
        """IP-16a-2 (D1 §2): one Intent per row-4 Decision this incarnation."""

        if (
            self._intent is None
            or not self._intent.enabled
            or not isinstance(value, _DecisionIssueResult)
            or value.status
            not in (DecisionIssueStatus.ISSUED, DecisionIssueStatus.EXISTING)
            or value.row is not SuccessorRow.ROW_4
            or value.decision_id is None
            or value.lineage_scope is None
            or value.decision_id in self._intended
        ):
            return
        self._maybe_intent_target(_IntentTarget(value.decision_id, value.lineage_scope))

    def _maybe_intent_target(self, target: _IntentTarget) -> None:
        if target.decision_id in self._intended:
            return
        self._intended[target.decision_id] = None
        _bound(self._intended)
        self._submit(_Stage.ACTION_INTENT, target)

    def _on_gate_open(self) -> None:
        """§3-6 item 2: on every transition into OPEN, re-submit each Intent
        target that met a closed gate and re-post ONE Policy scan with
        ``_policy_scans`` reset (so a flapping target cannot exhaust it): at
        most one re-submit round per OPEN transition, each chain still capped
        at ``MAX_POLICY_SCANS_PER_STARTUP`` links."""

        waiting = list(self._gate_waiting.values())
        self._gate_waiting.clear()
        for target in waiting:
            self._maybe_intent_target(target)
        if self._active and self._policy is not None and self._policy.enabled:
            self._policy_scans = 0
            self._policy_scan_posted = True
            self._maybe_policy_scan(None)

    def _wait_for_gate(self, target: _IntentTarget) -> None:
        self._intended.pop(target.decision_id, None)
        self._intent_retries.pop(target.decision_id, None)
        self._gate_waiting[target.decision_id] = target
        while len(self._gate_waiting) > OBSERVED_CAPACITY:
            del self._gate_waiting[next(iter(self._gate_waiting))]
        if self._gate is not None and self._gate.is_open():
            # The gate opened while this result was in flight: its OPEN
            # message was handled before this target was remembered.
            self._gate_waiting.pop(target.decision_id, None)
            self._maybe_intent_target(target)

    def _intent_done(self, target: _IntentTarget, value: object) -> None:
        if (
            isinstance(value, _IntentIssueResult)
            and value.reason is R18IntentReason.ADMISSION_GATE_CLOSED
        ):
            self._wait_for_gate(target)
            return
        if (
            isinstance(value, _IntentIssueResult)
            and value.status in (IntentIssueStatus.ISSUED, IntentIssueStatus.EXISTING)
            and value.intent_id is not None
        ):
            self._maybe_forecast(value.intent_id)
        if not (
            isinstance(value, _IntentIssueResult)
            and value.reason is R18IntentReason.IN_FLIGHT
        ):
            self._intent_retries.pop(target.decision_id, None)
            return
        attempt = self._intent_retries.get(target.decision_id, 0)
        if attempt >= len(RETRY_DELAYS_SECONDS):
            self._intent_retries.pop(target.decision_id, None)
            return
        self._intent_retries[target.decision_id] = attempt + 1
        while len(self._intent_retries) > OBSERVED_CAPACITY:
            del self._intent_retries[next(iter(self._intent_retries))]
        self._retries.append(
            _Retry(
                self._clock() + RETRY_DELAYS_SECONDS[attempt],
                0,
                _Stage.ACTION_INTENT,
                target,
            )
        )

    def _maybe_forecast(self, intent_id: str) -> None:
        """IP-16a-3 (D1 §2): one forecast request per Intent this incarnation."""

        if (
            self._expectation is None
            or not self._expectation.enabled
            or intent_id in self._forecasted
        ):
            return
        self._forecasted[intent_id] = None
        _bound(self._forecasted)
        self._submit(_Stage.EXPECTATION, _ForecastTarget(intent_id))

    def _maybe_scan(self) -> None:
        """At most ``MAX_SCANS_PER_STARTUP`` scans per driver incarnation."""

        if (
            self._expectation is None
            or not self._expectation.enabled
            or self._scans >= MAX_SCANS_PER_STARTUP
        ):
            return
        self._scans += 1
        self._submit(_Stage.EXPECTATION, _ForecastTarget(None))

    def _forecast_done(self, target: _ForecastTarget, value: object) -> None:
        if not isinstance(value, _ExpectationResult):
            return
        if (
            value.status
            in (ExpectationCommitStatus.COMMITTED, ExpectationCommitStatus.EXISTING)
            and value.intent_id is not None
        ):
            # IP-17a (D1 r7 §2): the Policy follows any forecast result.
            self._maybe_policy(value.intent_id)
        if target.intent_id is None:
            # The scan: only a COMMITTED continues it.
            if value.status is ExpectationCommitStatus.COMMITTED:
                self._maybe_scan()
            return
        if value.reason not in TRANSIENT_REASONS:
            self._forecast_retries.pop(target.intent_id, None)
            return
        attempt = self._forecast_retries.get(target.intent_id, 0)
        if attempt >= len(RETRY_DELAYS_SECONDS):
            self._forecast_retries.pop(target.intent_id, None)
            return
        self._forecast_retries[target.intent_id] = attempt + 1
        while len(self._forecast_retries) > OBSERVED_CAPACITY:
            del self._forecast_retries[next(iter(self._forecast_retries))]
        self._retries.append(
            _Retry(
                self._clock() + RETRY_DELAYS_SECONDS[attempt],
                0,
                _Stage.EXPECTATION,
                target,
            )
        )

    def _maybe_policy(self, intent_id: str) -> None:
        """IP-17a (D1 r7 §2): one Policy request per Intent this incarnation."""

        if self._policy is None or not self._policy.enabled or intent_id in self._policed:
            return
        self._policed[intent_id] = None
        _bound(self._policed)
        self._submit(_Stage.POLICY, _PolicyTarget(intent_id))

    def _maybe_policy_scan(self, after: str | None) -> None:
        """At most ``MAX_POLICY_SCANS_PER_STARTUP`` scans per driver incarnation."""

        if (
            self._policy is None
            or not self._policy.enabled
            or self._policy_scans >= MAX_POLICY_SCANS_PER_STARTUP
        ):
            return
        self._policy_scans += 1
        self._submit(_Stage.POLICY, _PolicyTarget(None, scan=True, after=after))

    def _policy_done(self, target: _PolicyTarget, value: object) -> None:
        if not isinstance(value, _PolicyResult):
            return
        if value.reason is R18PolicyReason.ADMISSION_GATE_CLOSED:
            # The gate opening re-posts one scan (§3-6); the target is
            # forgotten so a later forecast result can ask again.
            if target.intent_id is not None:
                self._policed.pop(target.intent_id, None)
                self._policy_retries.pop(target.intent_id, None)
            if (
                self._gate is not None
                and self._gate.is_open()
                and self._active
                and self._policy is not None
                and self._policy.enabled
            ):
                # Opened while the result was in flight (review P2-1).
                self._policy_scans = 0
                self._policy_scan_posted = True
                self._maybe_policy_scan(None)
            return
        if target.scan:
            # Chained on the result's intent id; a HOLD on one Intent is
            # skipped.  NONE_LEFT, R18_DISABLED or no id ends the chain.
            if (
                value.status is not PolicyCommitStatus.NONE_LEFT
                and value.reason is not R18PolicyReason.R18_DISABLED
                and value.intent_id is not None
            ):
                self._maybe_policy_scan(value.intent_id)
            return
        assert target.intent_id is not None
        if value.reason not in TRANSIENT_POLICY_REASONS:
            self._policy_retries.pop(target.intent_id, None)
            return
        attempt = self._policy_retries.get(target.intent_id, 0)
        if attempt >= len(RETRY_DELAYS_SECONDS):
            self._policy_retries.pop(target.intent_id, None)
            return
        self._policy_retries[target.intent_id] = attempt + 1
        while len(self._policy_retries) > OBSERVED_CAPACITY:
            del self._policy_retries[next(iter(self._policy_retries))]
        self._retries.append(
            _Retry(
                self._clock() + RETRY_DELAYS_SECONDS[attempt],
                0,
                _Stage.POLICY,
                target,
            )
        )

    # -- IP-19: Assessment -> Feedback ------------------------------------------

    def _on_feedback_scan(self) -> None:
        """A new chain (activation, or an Outcome was committed): the per-chain
        link counters restart, so a steady stream of Outcomes is never starved
        by an earlier chain that used its links."""

        if self._feedback is None or not self._feedback.enabled:
            return
        self._assess_scans = 0
        self._feedback_scans = 0
        self._maybe_assess_scan()

    def _feedback_backstop(self) -> None:
        """Re-run the chain about every ``FEEDBACK_BACKSTOP_SECONDS`` while active."""

        if (
            self._feedback is None
            or not self._feedback.enabled
            or not self._active
            or self._clock() < self._next_feedback_backstop
        ):
            return
        self._next_feedback_backstop = self._clock() + FEEDBACK_BACKSTOP_SECONDS
        self._on_feedback_scan()

    def _maybe_assess_scan(self) -> None:
        """At most ``MAX_FEEDBACK_SCANS_PER_STARTUP`` links per chain."""

        if (
            self._feedback is None
            or not self._feedback.enabled
            or self._assess_scans >= MAX_FEEDBACK_SCANS_PER_STARTUP
        ):
            return
        self._assess_scans += 1
        self._submit(_Stage.ASSESSMENT, _AssessmentTarget())

    def _assessment_done(self, value: object) -> None:
        if not isinstance(value, _AssessmentResult):
            return
        if value.status is AssessmentCommitStatus.EXISTING:
            self._cause_terminal()
        if value.status is AssessmentCommitStatus.COMMITTED:
            self._cause_terminal()
            self._scan_retries.pop(_Stage.ASSESSMENT.value, None)
            assert value.assessment is not None and value.assessment_id is not None
            if value.assessment.positive_feedback_eligible:
                self._maybe_feedback(value.assessment_id)
            self._maybe_assess_scan()
            return
        if self._retry_scan(_Stage.ASSESSMENT, _AssessmentTarget(), value.reason):
            return
        # NONE_LEFT or a HOLD ends the Assessment links; the Feedback scan
        # catches an Assessment whose Feedback never followed.
        self._maybe_feedback_scan()

    def _retry_scan(self, stage: _Stage, target: object, reason: object) -> bool:
        """A transient HOLD of a scan is retried with the bounded delays
        (review P2-1), so an Outcome is not left waiting for the next one."""

        key = stage.value
        if reason not in TRANSIENT_FEEDBACK_REASONS:
            self._scan_retries.pop(key, None)
            return False
        attempt = self._scan_retries.get(key, 0)
        if attempt >= len(RETRY_DELAYS_SECONDS):
            self._scan_retries.pop(key, None)
            return False
        self._scan_retries[key] = attempt + 1
        self._retries.append(_Retry(self._clock() + RETRY_DELAYS_SECONDS[attempt], 0, stage, target))
        return True

    def _maybe_feedback(self, assessment_id: str) -> None:
        """One Feedback request per Assessment this incarnation."""

        if (
            self._feedback is None
            or not self._feedback.enabled
            or assessment_id in self._feedbacked
        ):
            return
        self._feedbacked[assessment_id] = None
        _bound(self._feedbacked)
        self._submit(_Stage.FEEDBACK, _FeedbackTarget(assessment_id))

    def _maybe_feedback_scan(self) -> None:
        if (
            self._feedback is None
            or not self._feedback.enabled
            or self._feedback_scans >= MAX_FEEDBACK_SCANS_PER_STARTUP
        ):
            return
        self._feedback_scans += 1
        self._submit(_Stage.FEEDBACK, _FeedbackTarget(None))

    def _feedback_done(self, target: _FeedbackTarget, value: object) -> None:
        if not isinstance(value, _FeedbackResult):
            return
        if value.status in (FeedbackCommitStatus.COMMITTED, FeedbackCommitStatus.EXISTING):
            self._cause_terminal()
        if target.assessment_id is None:
            # The scan: only a COMMITTED continues it.
            if value.status is FeedbackCommitStatus.COMMITTED:
                self._maybe_feedback_scan()
            else:
                self._retry_scan(_Stage.FEEDBACK, target, value.reason)
            return
        if value.reason not in TRANSIENT_FEEDBACK_REASONS:
            self._feedback_retries.pop(target.assessment_id, None)
            return
        attempt = self._feedback_retries.get(target.assessment_id, 0)
        if attempt >= len(RETRY_DELAYS_SECONDS):
            self._feedback_retries.pop(target.assessment_id, None)
            return
        self._feedback_retries[target.assessment_id] = attempt + 1
        while len(self._feedback_retries) > OBSERVED_CAPACITY:
            del self._feedback_retries[next(iter(self._feedback_retries))]
        self._retries.append(
            _Retry(
                self._clock() + RETRY_DELAYS_SECONDS[attempt],
                0,
                _Stage.FEEDBACK,
                target,
            )
        )

    def _on_rescan(self, item: _Rescan) -> None:
        """#370 D0 §5: PROPOSED records left without a review, once each."""

        for goal_id in item.goal_ids:
            if goal_id not in self._reviewed:
                self._reviewed[goal_id] = None
                _bound(self._reviewed)
                self._submit(_Stage.GOAL_REVIEW, (None, goal_id))
        for commitment_id in item.commitment_ids:
            self._submit(_Stage.COMMITMENT_REVIEW, (None, commitment_id))
        for target in item.self_targets:
            self._maybe_self_propose(target)


def _bound(values: dict[Any, None]) -> None:
    """Keep incarnation-scoped sets bounded, evicting the oldest (P2-9/P2-b)."""
    while len(values) > OBSERVED_CAPACITY:
        del values[next(iter(values))]


def _result_code(value: object) -> str:
    if isinstance(value, R16OwnerOutcome):
        return value.status.value if value.reason is None else value.reason.value
    if isinstance(value, _DecisionIssueResult):
        # IP-10: a closed status and row ID only, never Decision contents.
        if value.row is not None:
            return f"{value.status.value}:{value.row.value}"
        if value.owner_reason is not None:
            return f"{value.status.value}:{value.owner_reason.value}"
        return value.status.value
    if isinstance(value, _IntentIssueResult):
        # IP-16a-2: a closed status and reason only, never Intent contents.
        if value.reason is not None:
            return f"{value.status.value}:{value.reason.value}"
        return value.status.value
    if isinstance(value, _ExpectationResult):
        # IP-16a-3: a closed status and reason only, never forecast contents.
        if value.reason is not None:
            return f"{value.status.value}:{value.reason.value}"
        return value.status.value
    if isinstance(value, _PolicyResult):
        # IP-17a: a closed status and reason only, never verdict contents.
        if value.reason is not None:
            return f"{value.status.value}:{value.reason.value}"
        return value.status.value
    if isinstance(value, (_AssessmentResult, _FeedbackResult)):
        # IP-19: a closed status and reason only, never record contents.
        if value.reason is not None:
            return f"{value.status.value}:{value.reason.value}"
        return value.status.value
    if isinstance(value, str) and value in {
        "interest_applied",
        "other_motivation_applied",
        "neutral_no_eligible_appraisal",
    }:
        return value
    return "other"


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


def self_goal_targets(
    motivation_state: object,
    goal_state: object,
    commitment_state: object,
    experience_ref: Callable[[str], _MotivationSourceRef | None],
    *,
    marker_sequence: int,
    min_strength: float,
    stale_after_hours: int,
    now: datetime,
) -> tuple[_SelfTarget, ...]:
    """IP-11 rescan (a): INTERESTs that may still ground a self Goal.

    Selection only, read by the server before the runtime starts; the goal
    owner re-checks every condition inside its own event.
    """

    from suzka.motivation.common import R13Reference, R13ReferenceKind
    from suzka.motivation.goal import goal_id_for_target
    from suzka.motivation.motivation import MotivationKind, MotivationLifecycle

    goals = {record.goal_id for record in getattr(goal_state, "records", ())}
    committed = {
        ref
        for record in getattr(commitment_state, "records", ())
        for ref in record.evidence_refs
    }
    ledger = {
        entry.motivation_id: entry.event_sequence
        for entry in getattr(motivation_state, "evidence_ledger", ())
    }
    found: list[_SelfTarget] = []
    for record in getattr(motivation_state, "records", ()):
        target = record.target
        if (
            record.kind is not MotivationKind.INTEREST
            or record.lifecycle is not MotivationLifecycle.ACTIVE
            or record.revision != 0
            or record.conflict_refs
            or record.strength < min_strength
            or not isinstance(target, R13Reference)
            or target.kind is not R13ReferenceKind.EXPERIENCE
            or ledger.get(record.motivation_id, 0) <= marker_sequence
            or goal_id_for_target(target) in goals
            or target in committed
            or now - record.revision_history[0].created_at
            > timedelta(hours=stale_after_hours)
        ):
            continue
        ref = experience_ref(target.reference)
        if ref is not None:
            found.append(_SelfTarget(target.reference, ref))
    return tuple(found)


def rescan_from(
    goal_state: object,
    commitment_state: object,
    self_targets: tuple[_SelfTarget, ...] = (),
) -> _Rescan:
    """PROPOSED user-request Goals and PROPOSED Commitments at startup."""

    from suzka.motivation.commitment import CommitmentLifecycle
    from suzka.motivation.common import R13ReferenceKind
    from suzka.motivation.goal import GoalLifecycle

    # IP-11: PROPOSED self-origin Goals are re-reviewed too (#379 D1 P1-5 b).
    goal_ids = tuple(
        record.goal_id
        for record in getattr(goal_state, "records", ())
        if record.lifecycle is GoalLifecycle.PROPOSED
        and any(
            ref.kind in (R13ReferenceKind.USER_REQUEST, R13ReferenceKind.MOTIVATION)
            for ref in record.origin_refs
        )
    )
    commitment_ids = tuple(
        record.commitment_id
        for record in getattr(commitment_state, "records", ())
        if record.lifecycle is CommitmentLifecycle.PROPOSED
    )
    return _Rescan(goal_ids, commitment_ids, self_targets)


__all__ = [
    "R16CommandKind",
    "R16CommandTicket",
    "R16Driver",
    "R16DriverReason",
    "R16TicketStatus",
    "R16TicketView",
    "parse_command",
    "rescan_from",
    "self_goal_targets",
]
