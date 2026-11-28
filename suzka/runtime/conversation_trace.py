"""ConversationTrace: the subject's short-term trace of recent Chat turns (#417 D1).

The trace holds references only (``episode_id`` into DB1), never turn text
(D1-a). Each item fades as ``R = exp(-Δt / S)`` in wall-clock time. A surprising
turn starts with a larger stability ``S`` (decision 6 (i)). Only a double recall
(the same episode was also retrieved from DB1) or the recall of a faint item
(``R < verbatim_above``) reinforces it, by ``gain × (1 − R)`` at most once per
turn; being shown alone never does (D1 v2 review P2-1). Items below
``forget_below`` are forgotten across all Contexts on every turn.

Timestamps are serialized as ``isoformat()`` (``+00:00``); AgentState must
carry the canonical value as an opaque dict, never as pydantic datetimes
(which would write ``Z`` and fail the canonical check).

This module is pure: it neither reads DB1 nor touches AgentState. The main
loop resolves text and decides when to mutate; AgentState captures
``ConversationTrace.state`` (#417 PR-B/PR-C).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import math
from threading import RLock

from suzka.identifiers import validate_identifier
from suzka.limits import (
    MAX_PERSISTED_EVENT_SEQUENCE,
    MAX_PERSISTED_REVISION,
    MAX_TRACE_CONTEXTS as MAX_TRACE_CONTEXTS,
    MAX_TRACE_ITEMS_PER_CONTEXT as MAX_TRACE_ITEMS_PER_CONTEXT,
    MAX_TRACE_ITEMS_TOTAL as MAX_TRACE_ITEMS_TOTAL,
    MAX_TRACE_STABILITY_HOURS as MAX_TRACE_STABILITY_HOURS,
)


MAX_TRACE_COUNTER = MAX_PERSISTED_REVISION

RECENT_CONVERSATION_HEADING = (
    "Recent conversation (as you remember it; faint items are shortened):"
)
FAINT_EXCERPT_CHARS = 60
COMMAND_TURN_TEXT = "(the user asked to review an earlier turn)"


class TraceTurnKind(str, Enum):
    CHAT = "chat"
    COMMAND = "command"


class TraceRemovalReason(str, Enum):
    FADED = "faded"
    CAPACITY = "capacity"
    CONTEXT_CAPACITY = "context_capacity"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class TraceParameters:
    """Validated tuning, normally built from ``ConversationTraceSettings``."""

    base_stability_hours: float = 1.0
    surprise_gain: float = 2.0
    recall_gain: float = 0.5
    double_recall_gain: float = 2.0
    max_stability_hours: float = 168.0
    forget_below: float = 0.2
    verbatim_above: float = 0.5
    max_items_per_context: int = 20
    max_rendered_bytes: int = 4096

    def __post_init__(self) -> None:
        for name in ("surprise_gain", "recall_gain", "double_recall_gain"):
            _finite(getattr(self, name), name, low=0.0)
        _finite(self.base_stability_hours, "base_stability_hours", low=0.0, open_low=True)
        _finite(
            self.max_stability_hours,
            "max_stability_hours",
            low=self.base_stability_hours,
            high=MAX_TRACE_STABILITY_HOURS,
        )
        _finite(self.forget_below, "forget_below", low=0.0, high=1.0, open_low=True)
        _finite(
            self.verbatim_above,
            "verbatim_above",
            low=self.forget_below,
            high=1.0,
            open_low=True,
            open_high=True,
        )
        if (
            type(self.max_items_per_context) is not int
            or not 1 <= self.max_items_per_context <= MAX_TRACE_ITEMS_PER_CONTEXT
        ):
            raise ValueError("max_items_per_context is out of range")
        if type(self.max_rendered_bytes) is not int or self.max_rendered_bytes < 1:
            raise ValueError("max_rendered_bytes is out of range")

    @classmethod
    def from_settings(cls, settings: object) -> TraceParameters:
        return cls(
            base_stability_hours=getattr(settings, "base_stability_hours"),
            surprise_gain=getattr(settings, "surprise_gain"),
            recall_gain=getattr(settings, "recall_gain"),
            double_recall_gain=getattr(settings, "double_recall_gain"),
            max_stability_hours=getattr(settings, "max_stability_hours"),
            forget_below=getattr(settings, "forget_below"),
            verbatim_above=getattr(settings, "verbatim_above"),
            max_items_per_context=getattr(settings, "max_items_per_context"),
            max_rendered_bytes=getattr(settings, "max_rendered_bytes"),
        )


@dataclass(frozen=True, slots=True)
class TraceItem:
    """One remembered turn: a DB1 reference and its retention state."""

    episode_id: str
    turn_kind: TraceTurnKind
    requested_at: datetime
    event_sequence: int
    surprise: float
    surprise_measured: bool
    stability_hours: float
    last_reinforced_at: datetime
    last_reinforced_event_sequence: int
    recall_count: int = 0
    double_recall_count: int = 0

    def __post_init__(self) -> None:
        validate_identifier(self.episode_id)
        if type(self.turn_kind) is not TraceTurnKind:
            raise TypeError("turn_kind must be a TraceTurnKind")
        _utc(self.requested_at, "requested_at")
        _utc(self.last_reinforced_at, "last_reinforced_at")
        if self.last_reinforced_at < self.requested_at:
            raise ValueError("an item cannot be reinforced before its turn")
        _bounded_int(self.event_sequence, "event_sequence", MAX_PERSISTED_EVENT_SEQUENCE)
        _finite(self.surprise, "surprise", low=0.0, high=1.0)
        if type(self.surprise_measured) is not bool:
            raise TypeError("surprise_measured must be a bool")
        if not self.surprise_measured and self.surprise != 0.0:
            raise ValueError("an unmeasured surprise must be 0")
        _finite(
            self.stability_hours,
            "stability_hours",
            low=0.0,
            high=MAX_TRACE_STABILITY_HOURS,
            open_low=True,
        )
        _bounded_int(
            self.last_reinforced_event_sequence,
            "last_reinforced_event_sequence",
            MAX_PERSISTED_EVENT_SEQUENCE,
        )
        if self.last_reinforced_event_sequence < self.event_sequence:
            raise ValueError("an item cannot be reinforced before its turn")
        _bounded_int(self.recall_count, "recall_count", MAX_TRACE_COUNTER)
        _bounded_int(self.double_recall_count, "double_recall_count", MAX_TRACE_COUNTER)

    def retention(self, now: datetime) -> float:
        """``R = exp(-Δt / S)`` with ``Δt = max(0, now − last_reinforced_at)``."""

        _utc(now, "now")
        elapsed_hours = max(timedelta(0), now - self.last_reinforced_at).total_seconds() / 3600
        return math.exp(-elapsed_hours / self.stability_hours)


@dataclass(frozen=True, slots=True)
class ContextTrace:
    context_id: str
    items: tuple[TraceItem, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.context_id)
        if type(self.items) is not tuple:
            raise TypeError("items must be a tuple")
        if not 1 <= len(self.items) <= MAX_TRACE_ITEMS_PER_CONTEXT:
            raise ValueError("a Context trace holds 1 to 32 items")
        episode_ids = [item.episode_id for item in self.items]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("a Context trace repeats an episode")
        order = [(item.requested_at, item.event_sequence) for item in self.items]
        if order != sorted(order):
            raise ValueError("Context trace items must be in turn order")


@dataclass(frozen=True, slots=True)
class ConversationTraceState:
    """Canonical, bounded, reference-only trace state (AgentState V11)."""

    revision: int = 0
    contexts: tuple[ContextTrace, ...] = ()

    def __post_init__(self) -> None:
        _bounded_int(self.revision, "revision", MAX_TRACE_COUNTER)
        if type(self.contexts) is not tuple:
            raise TypeError("contexts must be a tuple")
        if len(self.contexts) > MAX_TRACE_CONTEXTS:
            raise ValueError("too many Context traces")
        context_ids = [trace.context_id for trace in self.contexts]
        if context_ids != sorted(set(context_ids)):
            raise ValueError("Context traces must be unique and sorted")
        episode_ids = [item.episode_id for trace in self.contexts for item in trace.items]
        if len(episode_ids) > MAX_TRACE_ITEMS_TOTAL:
            raise ValueError("too many trace items")
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("an episode appears twice in the trace")

    @property
    def item_count(self) -> int:
        return sum(len(trace.items) for trace in self.contexts)

    def items_for(self, context_id: str) -> tuple[TraceItem, ...]:
        for trace in self.contexts:
            if trace.context_id == context_id:
                return trace.items
        return ()

    def canonical_value(self) -> dict[str, object]:
        """JSON-ready projection. Field names avoid the private-state key list."""

        return {
            "revision": self.revision,
            "contexts": [
                {
                    "context_id": trace.context_id,
                    "items": [_item_value(item) for item in trace.items],
                }
                for trace in self.contexts
            ],
        }

    @classmethod
    def from_value(cls, value: object) -> ConversationTraceState:
        if not isinstance(value, Mapping) or set(value) != {"revision", "contexts"}:
            raise ValueError("invalid ConversationTrace state")
        contexts = value["contexts"]
        if not isinstance(contexts, list):
            raise ValueError("invalid ConversationTrace contexts")
        traces = []
        for entry in contexts:
            if not isinstance(entry, Mapping) or set(entry) != {"context_id", "items"}:
                raise ValueError("invalid ConversationTrace context")
            items = entry["items"]
            if not isinstance(items, list):
                raise ValueError("invalid ConversationTrace items")
            traces.append(
                ContextTrace(entry["context_id"], tuple(_item_from_value(item) for item in items))
            )
        state = cls(revision=value["revision"], contexts=tuple(traces))
        if state.canonical_value() != value:
            raise ValueError("ConversationTrace state is not canonical")
        return state


@dataclass(frozen=True, slots=True)
class TraceRemoval:
    context_id: str
    episode_id: str
    reason: TraceRemovalReason


@dataclass(frozen=True, slots=True)
class RecalledTurn:
    """A trace item with the text resolved from DB1 for this turn only."""

    item: TraceItem
    retention: float
    user_text: str
    response_text: str


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One past exchange with its raw stored texts, for a chronological history."""

    episode_id: str
    user_text: str
    assistant_text: str


@dataclass(frozen=True, slots=True)
class RenderedTrace:
    text: str
    shown_episode_ids: tuple[str, ...]
    faint_episode_ids: tuple[str, ...]


class ConversationTrace:
    """Owner of the trace state; every mutation advances the revision once."""

    def __init__(self, parameters: TraceParameters, *, enabled: bool = False) -> None:
        if type(parameters) is not TraceParameters:
            raise TypeError("parameters must be TraceParameters")
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        self._parameters = parameters
        self._enabled = enabled
        self._restored_from_v11 = False
        self._state = ConversationTraceState()
        self._lock = RLock()

    @property
    def parameters(self) -> TraceParameters:
        return self._parameters

    @property
    def enabled(self) -> bool:
        """Configured on: the trace adds, recalls and forgets (D1 §1)."""

        return self._enabled

    @property
    def frozen(self) -> bool:
        """Disabled after V11 was committed: kept as is, never shown (D1-b)."""

        return not self._enabled and self._restored_from_v11

    @property
    def persists(self) -> bool:
        """Whether AgentState must carry the trace (V11): enabled, or already V11."""

        return self._enabled or self._restored_from_v11

    def restore_committed(self, state: ConversationTraceState | None) -> None:
        """Restore from a committed AgentState: V11 state, or None for older schemas.

        Called by ``AgentStateStore.restore_into`` at startup and on a failure
        restore, so a V10 committed snapshot clears the V11 flag (D1 v3 §4).
        """

        with self._lock:
            if state is None:
                self._state = ConversationTraceState()
                self._restored_from_v11 = False
            else:
                self.restore_exact(state)
                self._restored_from_v11 = True

    def committed_marker(self) -> tuple[ConversationTraceState, bool]:
        """State and V11 flag, for rolling back a failed restore."""

        with self._lock:
            return self._state, self._restored_from_v11

    def rollback_to(self, marker: tuple[ConversationTraceState, bool]) -> None:
        state, restored_from_v11 = marker
        with self._lock:
            self.restore_exact(state)
            self._restored_from_v11 = restored_from_v11

    @property
    def state(self) -> ConversationTraceState:
        with self._lock:
            return self._state

    def restore_exact(self, state: ConversationTraceState) -> None:
        """Replace the state as committed; nothing is forgotten here (D1 §4)."""

        if type(state) is not ConversationTraceState:
            raise TypeError("state must be a ConversationTraceState")
        with self._lock:
            self._state = state

    def candidates(self, context_id: str, now: datetime) -> tuple[tuple[TraceItem, float], ...]:
        """Items of one Context still above ``forget_below``, strongest first."""

        validate_identifier(context_id)
        with self._lock:
            scored = [(item, item.retention(now)) for item in self._state.items_for(context_id)]
        kept = [pair for pair in scored if pair[1] >= self._parameters.forget_below]
        return tuple(
            sorted(kept, key=lambda pair: (-pair[1], -pair[0].event_sequence, pair[0].episode_id))
        )

    def forget(self, now: datetime) -> tuple[TraceRemoval, ...]:
        """Forget faded items in every Context, then enforce the capacity bounds."""

        removals: list[TraceRemoval] = []
        with self._lock:
            traces: dict[str, list[TraceItem]] = {}
            for trace in self._state.contexts:
                kept = []
                for item in trace.items:
                    if item.retention(now) < self._parameters.forget_below:
                        removals.append(
                            TraceRemoval(trace.context_id, item.episode_id, TraceRemovalReason.FADED)
                        )
                    else:
                        kept.append(item)
                traces[trace.context_id] = kept
            removals.extend(self._enforce_capacity(traces, now))
            self._commit(traces, removals)
        return tuple(removals)

    def drop(self, context_id: str, episode_id: str) -> bool:
        """Remove an item whose DB1 episode is archived (named long-term effect)."""

        with self._lock:
            traces = {t.context_id: list(t.items) for t in self._state.contexts}
            items = traces.get(context_id, [])
            remaining = [item for item in items if item.episode_id != episode_id]
            if len(remaining) == len(items):
                return False
            traces[context_id] = remaining
            self._commit(
                traces, [TraceRemoval(context_id, episode_id, TraceRemovalReason.ARCHIVED)]
            )
        return True

    def reinforce(
        self,
        context_id: str,
        now: datetime,
        *,
        turn_sequence: int,
        shown_episode_ids: Iterable[str],
        double_recall_episode_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Reinforce faint recalls and double recalls of one turn; return reinforced ids.

        ``turn_sequence`` is the reinforcing CHAT event's sequence. An item is
        reinforced at most once per turn and never by the turn that added it
        (both keyed on that sequence, so a retry cannot double-reinforce).

        - Faint recall (shown while ``R < verbatim_above``): S grows by
          ``recall_gain × (1 − R)``, the decay clock restarts, ``recall_count``
          counts it.
        - Double recall (also retrieved from DB1): S grows by
          ``double_recall_gain × (1 − R)`` and ``double_recall_count`` counts
          it; the decay clock does not restart unless it is also a faint
          recall, so an item DB1 keeps retrieving still fades (#421 P3-2).
        """

        validate_identifier(context_id)
        _bounded_int(turn_sequence, "turn_sequence", MAX_PERSISTED_EVENT_SEQUENCE)
        shown = set(shown_episode_ids)
        double = set(double_recall_episode_ids)
        reinforced: list[str] = []
        with self._lock:
            traces = {t.context_id: list(t.items) for t in self._state.contexts}
            updated = []
            for item in traces.get(context_id, []):
                retention = item.retention(now)
                is_double = item.episode_id in double
                is_faint_recall = (
                    item.episode_id in shown and retention < self._parameters.verbatim_above
                )
                if (
                    item.event_sequence >= turn_sequence
                    or item.last_reinforced_event_sequence >= turn_sequence
                    or not (is_double or is_faint_recall)
                ):
                    updated.append(item)
                    continue
                gain = self._parameters.double_recall_gain if is_double else self._parameters.recall_gain
                stability = min(
                    item.stability_hours * (1.0 + gain * (1.0 - retention)),
                    self._parameters.max_stability_hours,
                )
                updated.append(
                    replace(
                        item,
                        stability_hours=stability,
                        last_reinforced_at=(
                            max(now, item.last_reinforced_at)
                            if is_faint_recall
                            else item.last_reinforced_at
                        ),
                        last_reinforced_event_sequence=turn_sequence,
                        recall_count=min(
                            item.recall_count + (1 if is_faint_recall else 0),
                            MAX_TRACE_COUNTER,
                        ),
                        double_recall_count=min(
                            item.double_recall_count + (1 if is_double else 0),
                            MAX_TRACE_COUNTER,
                        ),
                    )
                )
                reinforced.append(item.episode_id)
            if not reinforced:
                return ()
            traces[context_id] = updated
            self._commit(traces, [])
        return tuple(reinforced)

    def add(
        self,
        context_id: str,
        *,
        episode_id: str,
        turn_kind: TraceTurnKind,
        requested_at: datetime,
        event_sequence: int,
        surprise: float | None,
    ) -> tuple[TraceRemoval, ...]:
        """Add a committed turn; surprise None (invalid measurement) counts as 0."""

        validate_identifier(context_id)
        measured = surprise is not None
        value = 0.0 if surprise is None else min(max(float(surprise), 0.0), 1.0)
        parameters = self._parameters
        with self._lock:
            revision = self._state.revision + 1
            item = TraceItem(
                episode_id=episode_id,
                turn_kind=turn_kind,
                requested_at=requested_at,
                event_sequence=event_sequence,
                surprise=value,
                surprise_measured=measured,
                stability_hours=min(
                    parameters.base_stability_hours * (1.0 + parameters.surprise_gain * value),
                    parameters.max_stability_hours,
                ),
                last_reinforced_at=requested_at,
                # Not a reinforcement: the turn of addition never counts as a
                # recall (D1 §3), because reinforce skips sequences <= this.
                last_reinforced_event_sequence=event_sequence,
            )
            traces = {t.context_id: list(t.items) for t in self._state.contexts}
            if any(
                existing.episode_id == episode_id
                for others in traces.values()
                for existing in others
            ):
                raise ValueError("the episode is already in the trace")
            items = traces.setdefault(context_id, [])
            items.append(item)
            items.sort(key=lambda entry: (entry.requested_at, entry.event_sequence))
            removals = self._enforce_capacity(traces, requested_at)
            self._commit(traces, removals, revision=revision)
        return tuple(removals)

    def _enforce_capacity(
        self, traces: dict[str, list[TraceItem]], now: datetime
    ) -> list[TraceRemoval]:
        removals: list[TraceRemoval] = []
        per_context = self._parameters.max_items_per_context
        for context_id, items in traces.items():
            while len(items) > per_context:
                weakest = min(items, key=lambda item: (item.retention(now), item.event_sequence))
                items.remove(weakest)
                removals.append(
                    TraceRemoval(context_id, weakest.episode_id, TraceRemovalReason.CAPACITY)
                )
        while sum(len(items) for items in traces.values()) > MAX_TRACE_ITEMS_TOTAL:
            context_id, weakest = min(
                ((cid, item) for cid, items in traces.items() for item in items),
                key=lambda pair: (pair[1].retention(now), pair[1].event_sequence),
            )
            traces[context_id].remove(weakest)
            removals.append(
                TraceRemoval(context_id, weakest.episode_id, TraceRemovalReason.CAPACITY)
            )
        live = {cid: items for cid, items in traces.items() if items}
        while len(live) > MAX_TRACE_CONTEXTS:
            # The Context whose newest item is oldest goes first, frozen ones too.
            stale = min(
                live,
                key=lambda cid: (
                    max(item.requested_at for item in live[cid]),
                    cid,
                ),
            )
            for item in live.pop(stale):
                removals.append(
                    TraceRemoval(stale, item.episode_id, TraceRemovalReason.CONTEXT_CAPACITY)
                )
            traces[stale] = []
        return removals

    def _commit(
        self,
        traces: dict[str, list[TraceItem]],
        removals: Sequence[TraceRemoval],
        *,
        revision: int | None = None,
    ) -> None:
        contexts = tuple(
            ContextTrace(context_id, tuple(items))
            for context_id, items in sorted(traces.items())
            if items
        )
        if revision is None:
            if not removals and contexts == self._state.contexts:
                return
            revision = self._state.revision + 1
        self._state = ConversationTraceState(revision=revision, contexts=contexts)


def render_trace(
    recalled: Sequence[RecalledTurn],
    now: datetime,
    parameters: TraceParameters,
) -> RenderedTrace:
    """Pick by retention within the byte budget, then list oldest first (D1 §5)."""

    if not recalled:
        return RenderedTrace("", (), ())
    budget = parameters.max_rendered_bytes - len(RECENT_CONVERSATION_HEADING.encode()) - 1
    ordered = sorted(
        recalled,
        key=lambda turn: (-turn.retention, -turn.item.event_sequence, turn.item.episode_id),
    )
    chosen: list[tuple[RecalledTurn, str]] = []
    for turn in ordered:
        line = _render_line(turn, now, parameters, shorten=False)
        if len(line.encode()) + 1 > budget:
            # A long verbatim turn is shortened rather than hidden (#421 P3-1).
            line = _render_line(turn, now, parameters, shorten=True)
        size = len(line.encode()) + 1
        if size > budget:
            continue
        chosen.append((turn, line))
        budget -= size
    if not chosen:
        return RenderedTrace("", (), ())
    chosen.sort(key=lambda pair: (pair[0].item.requested_at, pair[0].item.event_sequence))
    text = "\n".join([RECENT_CONVERSATION_HEADING, *(line for _turn, line in chosen)])
    return RenderedTrace(
        text=text,
        shown_episode_ids=tuple(turn.item.episode_id for turn, _line in chosen),
        faint_episode_ids=tuple(
            turn.item.episode_id
            for turn, _line in chosen
            if turn.retention < parameters.verbatim_above
        ),
    )


def recent_history(recalled: Sequence[RecalledTurn], limit: int) -> tuple[HistoryEntry, ...]:
    """The most recent ``limit`` readable chat turns, oldest first.

    Order is ``(requested_at, event_sequence)`` (episode id breaks a remaining
    tie so the result is deterministic). COMMAND turns are left out. Texts are
    the raw stored texts: never the faint excerpt, never shortened. The
    caller decides which turns are readable; this function only orders,
    filters and cuts. Entry ``episode_id`` values give the episodes in the
    same order.
    """

    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a non-negative int")
    if limit == 0:
        return ()
    chat = [turn for turn in recalled if turn.item.turn_kind is TraceTurnKind.CHAT]
    chat.sort(
        key=lambda turn: (turn.item.requested_at, turn.item.event_sequence, turn.item.episode_id)
    )
    return tuple(
        HistoryEntry(turn.item.episode_id, turn.user_text, turn.response_text)
        for turn in chat[-limit:]
    )


def _render_line(
    turn: RecalledTurn, now: datetime, parameters: TraceParameters, *, shorten: bool
) -> str:
    age = _age_text(now - turn.item.requested_at)
    faint = turn.retention < parameters.verbatim_above
    label = f"{age} (faint)" if faint else f"{age} (shortened)" if shorten else age
    if turn.item.turn_kind is TraceTurnKind.COMMAND:
        return f"- {label} — {COMMAND_TURN_TEXT}"
    brief = faint or shorten
    user = _excerpt(turn.user_text) if brief else _one_line(turn.user_text)
    reply = _excerpt(turn.response_text) if brief else _one_line(turn.response_text)
    # JSON-quoted, like Working Memory evidence: past text cannot fake a new
    # line, a speaker label or the end of its own quote (#425 review P3-6).
    return f"- {label} — User: {_quote(user)} / You: {_quote(reply)}"


def _age_text(elapsed: timedelta) -> str:
    minutes = max(0, int(elapsed.total_seconds() // 60))
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"about {minutes} min ago"
    hours = minutes // 60
    if hours < 48:
        return f"about {hours} h ago"
    return f"about {hours // 24} days ago"


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _excerpt(text: str) -> str:
    line = _one_line(text)
    return line if len(line) <= FAINT_EXCERPT_CHARS else line[:FAINT_EXCERPT_CHARS] + "…"


def _item_value(item: TraceItem) -> dict[str, object]:
    return {
        "episode_id": item.episode_id,
        "turn_kind": item.turn_kind.value,
        "requested_at": item.requested_at.isoformat(),
        "event_sequence": item.event_sequence,
        "surprise": item.surprise,
        "surprise_measured": item.surprise_measured,
        "stability_hours": item.stability_hours,
        "last_reinforced_at": item.last_reinforced_at.isoformat(),
        "last_reinforced_event_sequence": item.last_reinforced_event_sequence,
        "recall_count": item.recall_count,
        "double_recall_count": item.double_recall_count,
    }


_ITEM_FIELDS = frozenset(
    {
        "episode_id",
        "turn_kind",
        "requested_at",
        "event_sequence",
        "surprise",
        "surprise_measured",
        "stability_hours",
        "last_reinforced_at",
        "last_reinforced_event_sequence",
        "recall_count",
        "double_recall_count",
    }
)


def _item_from_value(value: object) -> TraceItem:
    if not isinstance(value, Mapping) or set(value) != _ITEM_FIELDS:
        raise ValueError("invalid ConversationTrace item")
    try:
        return TraceItem(
            episode_id=value["episode_id"],
            turn_kind=TraceTurnKind(value["turn_kind"]),
            requested_at=_parse_utc(value["requested_at"]),
            event_sequence=value["event_sequence"],
            surprise=value["surprise"],
            surprise_measured=value["surprise_measured"],
            stability_hours=value["stability_hours"],
            last_reinforced_at=_parse_utc(value["last_reinforced_at"]),
            last_reinforced_event_sequence=value["last_reinforced_event_sequence"],
            recall_count=value["recall_count"],
            double_recall_count=value["double_recall_count"],
        )
    except (TypeError, KeyError) as error:
        raise ValueError("invalid ConversationTrace item") from error


def _parse_utc(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("timestamps must be ISO strings")
    parsed = datetime.fromisoformat(value)
    _utc(parsed, "timestamp")
    return parsed


def _utc(value: object, name: str) -> None:
    if type(value) is not datetime or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware UTC datetime")


def _bounded_int(value: object, name: str, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} is out of range")


def _finite(
    value: object,
    name: str,
    *,
    low: float | None = None,
    high: float | None = None,
    open_low: bool = False,
    open_high: bool = False,
) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if low is not None and (value < low or (open_low and value == low)):
        raise ValueError(f"{name} is out of range")
    if high is not None and (value > high or (open_high and value == high)):
        raise ValueError(f"{name} is out of range")


def conversation_trace_schema_max_bytes() -> int:
    """Worst-case canonical JSON bytes of a legal trace (AgentState V11 field).

    Built from a full-legal instance: every bound at its maximum, the longest
    identifiers, the longest float repr in range (23 characters), the largest
    integers, and microsecond UTC timestamps. AgentState serializes with sorted
    keys and compact separators, as here.
    """

    from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS

    longest_float = 2.2250738585072014e-308
    at = datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
    per_context = MAX_TRACE_ITEMS_TOTAL // MAX_TRACE_CONTEXTS
    contexts = []
    for c in range(MAX_TRACE_CONTEXTS):
        context_id = f"c{c:04d}".ljust(MAX_IDENTIFIER_CODEPOINTS, "x")
        items = tuple(
            TraceItem(
                episode_id=f"e{c:04d}{i:04d}".ljust(MAX_IDENTIFIER_CODEPOINTS, "x"),
                turn_kind=TraceTurnKind.COMMAND,
                requested_at=at,
                event_sequence=MAX_PERSISTED_EVENT_SEQUENCE,
                surprise=longest_float,
                surprise_measured=True,
                stability_hours=longest_float,
                last_reinforced_at=at,
                last_reinforced_event_sequence=MAX_PERSISTED_EVENT_SEQUENCE,
                recall_count=MAX_TRACE_COUNTER,
                double_recall_count=MAX_TRACE_COUNTER,
            )
            for i in range(per_context)
        )
        contexts.append(ContextTrace(context_id, items))
    state = ConversationTraceState(revision=MAX_TRACE_COUNTER, contexts=tuple(contexts))
    return len(
        json.dumps(
            state.canonical_value(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
