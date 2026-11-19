"""R19 IP-21a: the pure Inbox / Outbox functions (IP-21 D1 r3 §1, row IP-21a).

Nothing here reads a store, the Journal, a registry or the clock, and nothing
writes. IP-21b (the Outbox owner and one Scheduler step) and IP-21c (the ACK
owner and the Inbox routes) are thin adapters over these functions:

* ``TEMPLATE_TEXT``: the five fixed, code-owned message texts (A-9). The table
  is a closed constant, never built from data, with no format field.
* ``derive_inbox_view`` / ``InboxView``: the derived message state
  ``PENDING / ACKNOWLEDGED / REJECTED / EXPIRED`` from ``outbox_messages`` and
  ``outbox_acks`` rows only (§4-3), and the closed GET projection (§5-2). A state
  is never stored; a GET never changes anything.
* ``plan_emission`` / ``emission_candidates``: which closed Wakes still need a
  message, with the TTL anchored at record-byte lower bounds (§4-4, A-3), the
  handler's pure steps 3-6 (§4-5) and a closed ``OutboxHold`` vocabulary.
* ``ack_eligibility``: the read-only ACK pre-check (§5-4 steps 3-5), comparing
  an existing ACK by **status only** (``InboxAckV1.conflicts_with`` is not used).

``derive_wake_view`` and ``WakeEntry`` are untouched: Outbox / ACK records
stay invisible to the Wake ledger (§4-3, A-13). The only R17-R19 records this
module reads for a response are ``OUTBOX_MESSAGE`` and ``INBOX_ACK`` (A-14).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar

from suzka.r16.contracts import identifier
from suzka.r16.cycle_contracts import utc
from suzka.r16.history_records import RecordKind

if TYPE_CHECKING:
    from suzka.r16.cycle_contracts import CycleClosureV1
    from suzka.r16.history_codec import PolicyRow
    from suzka.r16.history_reader import AdoptedRecord

from suzka.r19.contracts import (
    MAX_MESSAGE_TTL,
    AckStatus,
    InboxAckV1,
    OutboxMessageV1,
    OutboxReason,
    OutboxTemplate,
    TerminalReason,
)
from suzka.r19.ledger import WakeEntry, WakeState, WakeView

_STAMP: Final = "%Y-%m-%dT%H:%M:%SZ"
_R = TypeVar("_R")

# D1 §4-4: ``L = lease_end - 120 s`` bounds the closing time from below only
# while a lease ends at most 120 s after its claim (``R19Settings.claim_lease_s``
# is capped at 120 and ``raw_state.CLAIM_LEASE_CAP_S`` is the same number; both
# are pinned by tests). If that cap is ever raised, this must follow it.
LEASE_BOUND_S: Final = 120
# D1 §5-1: the GET ``limit`` range (a constant, A-8).
MIN_LIST_LIMIT: Final = 1
MAX_LIST_LIMIT: Final = 50
# D1 §5-3 / A-11: the selector recorded when the client gave none.
DEFAULT_SESSION_SELECTOR: Final = "context-only"

# Times are not part of the dedupe key or the message id; the identity is built
# with these fixed values so it exists even for a message too old to write.
_PROBE_CREATED: Final = "2000-01-01T00:00:00Z"
_PROBE_EXPIRES: Final = "2000-01-01T00:00:01Z"

# --------------------------------------------------------------------- texts

# Proposed wording of D1 §5-2. The frontend may localize on (template, reason).
# Plain text: no HTML, no format field, no digit, no id, no URL.
TEMPLATE_TEXT: Final[Mapping[tuple[OutboxTemplate, OutboxReason], str]] = (
    MappingProxyType(
        {
            (
                OutboxTemplate.DECISION_ISSUED,
                OutboxReason.PLAN_SELECTED,
            ): "I looked at a goal again and made a new decision.",
            (
                OutboxTemplate.DECISION_ISSUED,
                OutboxReason.NO_ACTION,
            ): "I looked at a goal again and decided to take no action for now.",
            (
                OutboxTemplate.DECISION_CONFIRMED,
                OutboxReason.CONFIRMED,
            ): "I looked at a goal again and kept my earlier decision.",
            (
                OutboxTemplate.DECISION_HELD,
                OutboxReason.HELD,
            ): "I looked at a goal again and put the decision on hold.",
            (
                OutboxTemplate.CYCLE_FAILED,
                OutboxReason.FAILED,
            ): "I could not finish looking at a goal again.",
        }
    )
)


def message_text(template: OutboxTemplate, reason_code: OutboxReason) -> str:
    """The fixed text of a (template, reason code) pair; ``KeyError`` for a
    pair the DTO cannot hold (it checks the pairing at construction)."""

    return TEMPLATE_TEXT[(template, reason_code)]


# ------------------------------------------------------------- vocabularies


class MessageState(str, Enum):
    """The derived message state (D1 §4-3); there is no DELIVERED / READ."""

    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OutboxHold(str, Enum):
    """Why the Outbox owner (or the Scheduler's pre-filter) writes no message.

    A closed, source-free vocabulary. A hold never changes a Wake or a
    resolution; the next tick re-derives (D1 §4-5). ``NONE_LEFT`` is not a
    refusal: the message is already adopted (EXISTING)."""

    R19_DISABLED = "R19_DISABLED"
    HISTORY_UNAVAILABLE = "HISTORY_UNAVAILABLE"
    WAKE_UNKNOWN = "WAKE_UNKNOWN"
    WAKE_CORRUPT = "WAKE_CORRUPT"
    NOT_SETTLED = "NOT_SETTLED"
    NO_MESSAGE = "NO_MESSAGE"
    NONE_LEFT = "NONE_LEFT"
    IN_FLIGHT = "IN_FLIGHT"
    TOO_OLD = "TOO_OLD"
    CLOCK_UNVERIFIABLE = "CLOCK_UNVERIFIABLE"
    UNREPRESENTABLE = "UNREPRESENTABLE"
    HISTORY_CAPACITY = "HISTORY_CAPACITY"
    COLLISION = "COLLISION"
    QUEUE_DELAY = "QUEUE_DELAY"


# Permanent reasons: the same input gives the same answer, so the pacer sticks
# them for the process lifetime (D1 §4-5, last paragraph).
PERMANENT_OUTBOX_HOLDS: Final = frozenset(
    {OutboxHold.TOO_OLD, OutboxHold.UNREPRESENTABLE}
)


class AckResult(str, Enum):
    """The ACK event's outcome (D1 §5-4 step 6)."""

    COMMITTED = "COMMITTED"
    EXISTING = "EXISTING"
    CONFLICT = "CONFLICT"
    EXPIRED = "EXPIRED"
    NOT_FOUND = "NOT_FOUND"
    HOLD = "HOLD"


class AckHold(str, Enum):
    """The closed reasons of a ``HOLD`` ACK result (a 503 or 409 body code)."""

    R19_DISABLED = "R19_DISABLED"
    HISTORY_UNAVAILABLE = "HISTORY_UNAVAILABLE"
    IN_FLIGHT = "IN_FLIGHT"
    CONTEXT_CONFLICT = "CONTEXT_CONFLICT"
    CLOCK_UNVERIFIABLE = "CLOCK_UNVERIFIABLE"
    HISTORY_CAPACITY = "HISTORY_CAPACITY"
    COLLISION = "COLLISION"


class AckEligibility(str, Enum):
    """The read-only pre-check's answer; only ``WRITE`` submits an event."""

    WRITE = "WRITE"
    EXISTING = "EXISTING"
    CONFLICT = "CONFLICT"
    EXPIRED = "EXPIRED"
    NOT_FOUND = "NOT_FOUND"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class AckCheck:
    eligibility: AckEligibility
    hold: AckHold | None = None

    def __post_init__(self) -> None:
        if type(self.eligibility) is not AckEligibility:
            raise TypeError("eligibility must be an AckEligibility")
        if (self.eligibility is AckEligibility.HOLD) != (self.hold is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.hold is not None and type(self.hold) is not AckHold:
            raise TypeError("hold must be an AckHold")


# ---------------------------------------------------------------- the view


def _expect(value: object, kind: type[_R]) -> _R:
    if type(value) is not kind:
        raise TypeError(f"an adopted record must be a {kind.__name__}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class InboxEntry:
    """One message, its ACK row (if any) and the derived state."""

    message: OutboxMessageV1
    ack: InboxAckV1 | None
    state: MessageState

    @property
    def message_id(self) -> str:
        return self.message.message_id


@dataclass(frozen=True, slots=True)
class InboxItem:
    """The closed GET projection of one message (D1 §5-2): the only thing a
    client learns. No source id, digest, dedupe key, incarnation, Wake / cycle /
    Decision id, cause, policy row, Context id or ACK event field is here."""

    message_id: str
    template: OutboxTemplate
    reason_code: OutboxReason
    text: str
    created_utc: str
    expires_utc: str
    state: MessageState
    acked_utc: str | None

    def public_value(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "template": self.template.value,
            "reason_code": self.reason_code.value,
            "text": self.text,
            "created_utc": self.created_utc,
            "expires_utc": self.expires_utc,
            "state": self.state.value,
            "acked_utc": self.acked_utc,
        }


@dataclass(frozen=True, slots=True)
class InboxView:
    """The messages of exactly one Context at ``as_of_utc``, newest first."""

    as_of_utc: str
    context_id: str
    entries: tuple[InboxEntry, ...]
    # Records that tie to no message of the input (kind, id): logged, never shown.
    orphans: tuple[tuple[RecordKind, str], ...] = ()

    def get(self, message_id: str) -> InboxEntry | None:
        return next(
            (item for item in self.entries if item.message_id == message_id), None
        )

    def items(self, limit: int = MAX_LIST_LIMIT) -> tuple[InboxItem, ...]:
        if (
            type(limit) is not int
            or isinstance(limit, bool)
            or not MIN_LIST_LIMIT <= limit <= MAX_LIST_LIMIT
        ):
            raise ValueError("limit must be an integer from 1 to 50")
        return tuple(project(item) for item in self.entries[:limit])

    def public_value(self, limit: int = MAX_LIST_LIMIT) -> dict[str, object]:
        """The GET body of D1 §5-2: exactly ``messages`` and ``as_of_utc``."""

        return {
            "messages": [item.public_value() for item in self.items(limit)],
            "as_of_utc": self.as_of_utc,
        }


def project(entry: InboxEntry) -> InboxItem:
    """The closed projection of one entry; built from the two DTOs only."""

    message = entry.message
    return InboxItem(
        message.message_id,
        message.template,
        message.reason_code,
        message_text(message.template, message.reason_code),
        message.created_utc,
        message.expires_utc,
        entry.state,
        None if entry.ack is None else entry.ack.occurred_utc,
    )


def _state(message: OutboxMessageV1, ack: InboxAckV1 | None, now: datetime) -> MessageState:
    if ack is not None:
        return (
            MessageState.ACKNOWLEDGED
            if ack.status is AckStatus.ACKNOWLEDGED
            else MessageState.REJECTED
        )
    if now >= utc(message.expires_utc, "expires_utc"):
        return MessageState.EXPIRED
    return MessageState.PENDING


def derive_inbox_view(
    records: Iterable[AdoptedRecord], now_utc: str, *, context_id: str
) -> InboxView:
    """The Inbox of ``context_id`` at ``now_utc`` over adopted records.

    Pure and total over well-typed input. It matches ``context_id`` by string
    equality and reads no registry (D1 §5-3: no eligibility check, no
    existence oracle). Every record other than ``OUTBOX_MESSAGE`` /
    ``INBOX_ACK`` is ignored. An ACK ties to its message only when the message
    id and the Context id both match (the trigger guarantees it; a record that
    does not is an orphan, never an ACK of that message). Raises ``TypeError``
    when an adopted record is not the exact type its kind names (a decoding
    error, never a state) and ``ValueError`` for a bad instant or Context id.
    """

    now = utc(now_utc, "now_utc")
    identifier(context_id, "context_id")
    messages: dict[str, OutboxMessageV1] = {}
    acks: list[InboxAckV1] = []
    orphans: list[tuple[RecordKind, str]] = []
    for item in records:
        kind = item.record_kind
        if kind is RecordKind.OUTBOX_MESSAGE:
            message = _expect(item.record, OutboxMessageV1)
            if message.message_id in messages:
                orphans.append((kind, message.message_id))
                continue
            messages[message.message_id] = message
        elif kind is RecordKind.INBOX_ACK:
            ack = _expect(item.record, InboxAckV1)
            acks.append(ack)
    ack_of: dict[str, InboxAckV1] = {}
    for ack in acks:
        owner = messages.get(ack.message_id)
        if (
            owner is None
            or owner.context_id != ack.context_id
            or ack.message_id in ack_of
        ):
            orphans.append((RecordKind.INBOX_ACK, ack.ack_id))
            continue
        ack_of[ack.message_id] = ack
    chosen = [
        InboxEntry(message, ack_of.get(key), _state(message, ack_of.get(key), now))
        for key, message in messages.items()
        if message.context_id == context_id
    ]
    chosen.sort(key=lambda item: (item.message.created_utc, item.message_id), reverse=True)
    return InboxView(now_utc, context_id, tuple(chosen), tuple(orphans))


def adopted_message_ids(records: Iterable[AdoptedRecord]) -> frozenset[str]:
    """Every adopted message id (any Context): what "already adopted" means for
    the emitter's dedupe step (D1 §4-5 step 4)."""

    found: set[str] = set()
    for item in records:
        if item.record_kind is RecordKind.OUTBOX_MESSAGE:
            message = _expect(item.record, OutboxMessageV1)
            found.add(message.message_id)
    return frozenset(found)


# --------------------------------------------------------------------- ACK


def ack_selector(client_session_id: str | None) -> str:
    """The ACK row's ``session_selector`` (D1 §5-3, A-11): the client's own id
    when given, else the closed literal ``context-only``."""

    if client_session_id is None:
        return DEFAULT_SESSION_SELECTOR
    identifier(client_session_id, "client_session_id")
    return client_session_id


def same_ack_status(existing: InboxAckV1, status: AckStatus) -> bool:
    """The replay comparison (D1 §5-4, P2-1): **status only**. The Context
    already matched, and a replay with another or no ``client_session_id`` is
    still the same ACK; the stored selector is never rewritten.
    ``InboxAckV1.conflicts_with`` is deliberately not used."""

    if type(existing) is not InboxAckV1:
        raise TypeError("existing must be an InboxAckV1")
    if type(status) is not AckStatus:
        raise TypeError("status must be an AckStatus")
    return existing.status is status


def ack_eligibility(
    view: InboxView,
    message_id: str,
    status: AckStatus,
    now_utc: str,
    *,
    ack_key_live: bool = False,
    clock_floor: str | None = None,
) -> AckCheck:
    """The read-only ACK pre-check (D1 §5-4 steps 3-5); never an event.

    ``view`` is the Inbox of the request's Context, so a message of another
    Context and an unknown one are the same ``NOT_FOUND``. In order: an unknown
    message; an adopted ACK (same status: ``EXISTING``; the other status:
    ``CONFLICT``, the original is never rewritten); an unacknowledged message
    at or past its expiry: ``EXPIRED``; a PREPARED ACK not yet adopted
    (``ack_key_live``): ``HOLD IN_FLIGHT``; an instant before the message was
    created or before ``clock_floor``: ``HOLD CLOCK_UNVERIFIABLE``; else
    ``WRITE``. An ACKed message that has since expired still answers its
    replay with ``EXISTING`` / ``CONFLICT`` (the ACK row is immutable)."""

    if type(status) is not AckStatus:
        raise TypeError("status must be an AckStatus")
    if type(ack_key_live) is not bool:
        raise TypeError("ack_key_live must be a bool")
    now = utc(now_utc, "now_utc")
    entry = view.get(message_id) if type(message_id) is str else None
    if entry is None:
        return AckCheck(AckEligibility.NOT_FOUND)
    if entry.ack is not None:
        if same_ack_status(entry.ack, status):
            return AckCheck(AckEligibility.EXISTING)
        return AckCheck(AckEligibility.CONFLICT)
    if now >= utc(entry.message.expires_utc, "expires_utc"):
        return AckCheck(AckEligibility.EXPIRED)
    if ack_key_live:
        return AckCheck(AckEligibility.HOLD, AckHold.IN_FLIGHT)
    floor = None if clock_floor is None else utc(clock_floor, "clock_floor")
    if now < utc(entry.message.created_utc, "created_utc") or (
        floor is not None and now < floor
    ):
        return AckCheck(AckEligibility.HOLD, AckHold.CLOCK_UNVERIFIABLE)
    return AckCheck(AckEligibility.WRITE)


# ---------------------------------------------------------------- emission


@dataclass(frozen=True, slots=True)
class EmissionPlan:
    """What the emitter would do for one Wake at one instant (D1 §4-5)."""

    wakeup_id: str
    state: WakeState
    # None only when no identity could be derived (see ``hold``).
    message_id: str | None
    # ``L``: a lower bound of the closing time, from record bytes (§4-4).
    lower_bound_utc: str | None
    message: OutboxMessageV1 | None
    hold: OutboxHold | None

    def __post_init__(self) -> None:
        if (self.message is None) == (self.hold is None):
            raise ValueError("exactly one of a message and a hold")
        if self.hold is not None and type(self.hold) is not OutboxHold:
            raise TypeError("hold must be an OutboxHold")
        if self.message is not None and self.message.message_id != self.message_id:
            raise ValueError("the message id must be the message's")

    @property
    def ready(self) -> bool:
        """A message can be written now (the pre-filter passes)."""
        return self.message is not None


def decision_row_of(closure: CycleClosureV1) -> PolicyRow | None:
    """The issued Decision's ``policy_row`` (``closure.result.policy_row``) for a
    DECISION_ISSUED closure; None for a HOLD or EXISTING receipt."""

    decision = closure.decision
    return None if decision is None else decision.policy_row


def _build(
    entry: WakeEntry, created_utc: str, expires_utc: str
) -> OutboxMessageV1:
    """The DTO builder for a settled entry (a ``ValueError`` / ``TypeError``
    for what the builders refuse)."""

    if entry.state is WakeState.CLOSED:
        closure = entry.closure
        if closure is None:
            raise ValueError("a closed Wake has its closure")
        return OutboxMessageV1.for_completion(
            entry.wake,
            closure.resolution,
            decision_row_of(closure),
            created_utc=created_utc,
            expires_utc=expires_utc,
        )
    terminal = entry.terminal
    if terminal is None:
        raise ValueError("a failed Wake has its terminal")
    return OutboxMessageV1.for_failed_cycle(
        entry.wake, terminal, created_utc=created_utc, expires_utc=expires_utc
    )


def closing_lower_bound(entry: WakeEntry) -> datetime | None:
    """``L``, a lower bound of the time the cycle closed, from record bytes only
    (D1 §4-4, P2-3; the Journal is not read, so there is no "unreadable" case).

    * a resolution, or an ATTEMPTS_EXHAUSTED terminal: the latest claim's
      ``claim_lease_expires_utc - 120 s`` (the result is written under that
      claim, and a lease ends at most 120 s after it; see ``LEASE_BOUND_S``);
    * an EXPIRED_WAKE terminal: the Wake's expiry (the terminal needs ``T``
      past it).

    A bound can only make a message expire earlier than its true 7 days, never
    later. None when the records do not give one (no claim)."""

    terminal = entry.terminal
    if terminal is not None and terminal.reason is TerminalReason.EXPIRED_WAKE:
        return utc(entry.expiry_utc, "expiry_utc")
    claim = entry.latest_claim
    if claim is None:
        return None
    return utc(claim.claim_lease_expires_utc, "claim_lease_expires_utc") - timedelta(
        seconds=LEASE_BOUND_S
    )


def _stamp(value: datetime) -> str:
    return value.strftime(_STAMP)


def plan_emission(
    entry: WakeEntry,
    now_utc: str,
    *,
    outbox_ttl_s: int,
    adopted_ids: frozenset[str] = frozenset(),
    clock_floor: str | None = None,
) -> EmissionPlan:
    """The pure part of the handler order of D1 §4-5 (steps 3 to 6) for one entry.

    ``adopted_ids`` are the adopted message ids (see ``adopted_message_ids``);
    ``clock_floor`` the high-water mark the caller computed (None: no check).

    * CORRUPT: ``WAKE_CORRUPT``; CANCELLED: ``NO_MESSAGE`` (a superseded Wake
      has no result, A-15); any other unsettled state: ``NOT_SETTLED``.
    * the identity (Context, incarnation, result digest, source id, template;
      no time) is built first; a refusal of the builders or a Context that is
      not the Wake's: ``UNREPRESENTABLE``;
    * **dedupe first**: an adopted id: ``NONE_LEFT`` (EXISTING, nothing written);
    * ``created_utc = T`` (``now_utc``), ``expires_utc = min(T + ttl, L + 7 d)``;
      ``T >= expires_utc``: ``TOO_OLD``; ``T < L`` or ``T < clock_floor``:
      ``CLOCK_UNVERIFIABLE``; no ``L``: ``UNREPRESENTABLE``.
    """

    if type(entry) is not WakeEntry:
        raise TypeError("entry must be a WakeEntry")
    now = utc(now_utc, "now_utc")
    if (
        type(outbox_ttl_s) is not int
        or isinstance(outbox_ttl_s, bool)
        or not 0 < outbox_ttl_s <= int(MAX_MESSAGE_TTL.total_seconds())
    ):
        raise ValueError("outbox_ttl_s must be an integer from 1 to 7 days")
    floor = None if clock_floor is None else utc(clock_floor, "clock_floor")
    base = (entry.wakeup_id, entry.state)

    def held(
        hold: OutboxHold,
        message_id: str | None = None,
        lower: datetime | None = None,
    ) -> EmissionPlan:
        return EmissionPlan(
            *base, message_id, None if lower is None else _stamp(lower), None, hold
        )

    if entry.state is WakeState.CORRUPT:
        return held(OutboxHold.WAKE_CORRUPT)
    if entry.state is WakeState.CANCELLED:
        return held(OutboxHold.NO_MESSAGE)
    if entry.state not in (WakeState.CLOSED, WakeState.FAILED):
        return held(OutboxHold.NOT_SETTLED)
    try:
        identity = _build(entry, _PROBE_CREATED, _PROBE_EXPIRES)
    except (ValueError, TypeError):
        return held(OutboxHold.UNREPRESENTABLE)
    if identity.context_id != entry.wake.context_id:
        return held(OutboxHold.UNREPRESENTABLE, identity.message_id)
    if identity.message_id in adopted_ids:
        return held(OutboxHold.NONE_LEFT, identity.message_id)
    lower = closing_lower_bound(entry)
    if lower is None:
        return held(OutboxHold.UNREPRESENTABLE, identity.message_id)
    expires = min(now + timedelta(seconds=outbox_ttl_s), lower + MAX_MESSAGE_TTL)
    if now >= expires:
        return held(OutboxHold.TOO_OLD, identity.message_id, lower)
    if now < lower or (floor is not None and now < floor):
        return held(OutboxHold.CLOCK_UNVERIFIABLE, identity.message_id, lower)
    try:
        message = _build(entry, _stamp(now), _stamp(expires))
    except (ValueError, TypeError):
        return held(OutboxHold.UNREPRESENTABLE, identity.message_id, lower)
    return EmissionPlan(
        *base, message.message_id, _stamp(lower), message, None
    )


def emission_candidates(
    view: WakeView, adopted_ids: frozenset[str], *, outbox_ttl_s: int
) -> tuple[EmissionPlan, ...]:
    """The derived candidate list of D1 §4-5, in entry order.

    One plan for every ``CLOSED`` / ``FAILED`` Wake of ``view`` whose message
    is not adopted yet (a CANCELLED, CORRUPT or unsettled Wake is not a
    candidate; an adopted message is skipped). The caller submits the first
    ``ready`` plan; the others carry the hold the pacer accounts for. There is
    no periodic "complete scan" event: this is a function of the records and the
    instant (``view.as_of_utc``), with ``view.clock_floor_utc`` as the floor."""

    plans = []
    for entry in view.entries:
        if entry.state not in (WakeState.CLOSED, WakeState.FAILED):
            continue
        plan = plan_emission(
            entry,
            view.as_of_utc,
            outbox_ttl_s=outbox_ttl_s,
            adopted_ids=adopted_ids,
            clock_floor=view.clock_floor_utc,
        )
        if plan.hold is OutboxHold.NONE_LEFT:
            continue
        plans.append(plan)
    return tuple(plans)


__all__ = [
    "AckCheck",
    "AckEligibility",
    "AckHold",
    "AckResult",
    "DEFAULT_SESSION_SELECTOR",
    "EmissionPlan",
    "InboxEntry",
    "InboxItem",
    "InboxView",
    "LEASE_BOUND_S",
    "MAX_LIST_LIMIT",
    "MIN_LIST_LIMIT",
    "MessageState",
    "OutboxHold",
    "PERMANENT_OUTBOX_HOLDS",
    "TEMPLATE_TEXT",
    "ack_eligibility",
    "ack_selector",
    "adopted_message_ids",
    "closing_lower_bound",
    "decision_row_of",
    "derive_inbox_view",
    "emission_candidates",
    "message_text",
    "plan_emission",
    "project",
    "same_ack_status",
]
