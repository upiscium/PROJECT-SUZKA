"""#417 PR-A: pure ConversationTrace retention, reinforcement, capacity and rendering."""

from datetime import datetime, timedelta, timezone
import math

import pytest
from pydantic import ValidationError

from suzka.config.schema import ConversationTraceSettings
from suzka.r16.commands import R16CommandKind, parse_command
from suzka.r16.driver import parse_command as driver_parse_command
from suzka.runtime.conversation_trace import (
    COMMAND_TURN_TEXT,
    MAX_TRACE_CONTEXTS,
    MAX_TRACE_ITEMS_TOTAL,
    RECENT_CONVERSATION_HEADING,
    ContextTrace,
    ConversationTrace,
    ConversationTraceState,
    RecalledTurn,
    TraceItem,
    TraceParameters,
    TraceRemovalReason,
    TraceTurnKind,
    render_trace,
)

T0 = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
CTX = "conversation.default"


def _trace(**overrides: object) -> ConversationTrace:
    return ConversationTrace(TraceParameters(**overrides))  # type: ignore[arg-type]


def _add(trace: ConversationTrace, n: int, *, at: datetime = T0, surprise: float | None = 0.0,
         context: str = CTX, kind: TraceTurnKind = TraceTurnKind.CHAT) -> str:
    episode = f"episode-{context[-4:]}-{n:04d}"
    trace.add(context, episode_id=episode, turn_kind=kind, requested_at=at,
              event_sequence=n, surprise=surprise)
    return episode


def test_retention_follows_exp_decay_in_wall_clock_time() -> None:
    trace = _trace()
    _add(trace, 1)
    [item] = trace.state.items_for(CTX)
    assert item.stability_hours == 1.0
    assert item.retention(T0) == 1.0
    assert item.retention(T0 + timedelta(hours=1)) == pytest.approx(math.exp(-1))
    # A clock that went backwards never raises retention above 1.
    assert item.retention(T0 - timedelta(hours=3)) == 1.0


def test_surprise_slows_forgetting_and_unmeasured_surprise_counts_as_zero() -> None:
    trace = _trace()
    calm = _add(trace, 1, surprise=0.0)
    surprising = _add(trace, 2, surprise=1.0)
    unmeasured = _add(trace, 3, surprise=None)
    items = {item.episode_id: item for item in trace.state.items_for(CTX)}
    assert items[calm].stability_hours == 1.0
    assert items[surprising].stability_hours == 3.0
    assert items[unmeasured].stability_hours == 1.0
    assert items[unmeasured].surprise_measured is False


def test_faded_items_are_forgotten_in_every_context() -> None:
    trace = _trace()
    _add(trace, 1, context="conversation.a")
    _add(trace, 2, context="conversation.b", surprise=1.0)
    later = T0 + timedelta(hours=2)  # R = e^-2 ≈ 0.135 for S=1, e^-2/3 ≈ 0.51 for S=3
    removals = trace.forget(later)
    assert [(r.context_id, r.reason) for r in removals] == [
        ("conversation.a", TraceRemovalReason.FADED)
    ]
    assert trace.state.items_for("conversation.a") == ()
    assert len(trace.state.items_for("conversation.b")) == 1


def test_being_shown_alone_never_reinforces() -> None:
    trace = _trace()
    episode = _add(trace, 1)
    now = T0 + timedelta(minutes=10)  # R ≈ 0.85, above verbatim_above
    assert trace.reinforce(CTX, now, shown_episode_ids=[episode], double_recall_episode_ids=[]) == ()
    [item] = trace.state.items_for(CTX)
    assert item.last_reinforced_at == T0
    assert item.recall_count == 0


def test_faint_recall_reinforces_by_spacing() -> None:
    trace = _trace()
    episode = _add(trace, 1)
    now = T0 + timedelta(hours=1)  # R = e^-1 ≈ 0.37 < 0.5
    assert trace.reinforce(CTX, now, shown_episode_ids=[episode], double_recall_episode_ids=[]) == (episode,)
    [item] = trace.state.items_for(CTX)
    assert item.stability_hours == pytest.approx(1.0 * (1 + 0.5 * (1 - math.exp(-1))))
    assert item.last_reinforced_at == now
    assert item.recall_count == 1
    assert item.double_recall_count == 0


def test_double_recall_reinforces_more_and_only_once_per_turn() -> None:
    trace = _trace()
    episode = _add(trace, 1)
    now = T0 + timedelta(minutes=30)
    retention = math.exp(-0.5)
    trace.reinforce(CTX, now, shown_episode_ids=[], double_recall_episode_ids=[episode])
    [item] = trace.state.items_for(CTX)
    assert item.stability_hours == pytest.approx(1.0 * (1 + 2.0 * (1 - retention)))
    assert item.double_recall_count == 1
    # The turn that added an item does not count it as recalled.
    fresh = _add(trace, 2, at=now)
    assert trace.reinforce(CTX, now, shown_episode_ids=[], double_recall_episode_ids=[fresh]) == (fresh,)


def test_the_turn_of_addition_is_not_a_recall() -> None:
    trace = _trace()
    revision_before = trace.state.revision
    _add(trace, 1)
    [item] = trace.state.items_for(CTX)
    assert item.last_reinforced_revision == revision_before + 1 == trace.state.revision


def test_stability_is_capped() -> None:
    trace = _trace(base_stability_hours=100.0, max_stability_hours=168.0, surprise_gain=2.0)
    _add(trace, 1, surprise=1.0)
    [item] = trace.state.items_for(CTX)
    assert item.stability_hours == 168.0


def test_per_context_capacity_drops_the_weakest() -> None:
    trace = _trace(max_items_per_context=2)
    first = _add(trace, 1, at=T0)
    _add(trace, 2, at=T0 + timedelta(minutes=1), surprise=1.0)
    removals = trace.add(CTX, episode_id="episode-0003", turn_kind=TraceTurnKind.CHAT,
                         requested_at=T0 + timedelta(minutes=2), event_sequence=3, surprise=0.0)
    assert [(r.episode_id, r.reason) for r in removals] == [(first, TraceRemovalReason.CAPACITY)]


def test_context_count_is_bounded_by_staleness() -> None:
    trace = _trace()
    for n in range(MAX_TRACE_CONTEXTS + 1):
        _add(trace, n, context=f"conversation.c{n:04d}", at=T0 + timedelta(seconds=n))
    assert len(trace.state.contexts) == MAX_TRACE_CONTEXTS
    assert trace.state.items_for("conversation.c0000") == ()


def test_archived_items_can_be_dropped_by_name() -> None:
    trace = _trace()
    episode = _add(trace, 1)
    assert trace.drop(CTX, episode) is True
    assert trace.drop(CTX, episode) is False
    assert trace.state.contexts == ()


def test_restore_never_forgets_and_state_round_trips() -> None:
    trace = _trace()
    _add(trace, 1, surprise=0.4)
    _add(trace, 2, context="conversation.b")
    state = trace.state
    value = state.canonical_value()
    assert ConversationTraceState.from_value(value) == state
    restored = _trace()
    restored.restore_exact(state)
    assert restored.state == state
    with pytest.raises(ValueError):
        ConversationTraceState.from_value({**value, "extra": 1})


def test_state_rejects_text_like_fields_and_bad_shapes() -> None:
    trace = _trace()
    _add(trace, 1)
    value = trace.state.canonical_value()
    item = value["contexts"][0]["items"][0]  # type: ignore[index]
    with pytest.raises(ValueError):
        ConversationTraceState.from_value(
            {**value, "contexts": [{"context_id": CTX, "items": [{**item, "content": "hi"}]}]}
        )
    with pytest.raises(ValueError):
        TraceItem(**{**_item_kwargs(), "requested_at": datetime(2026, 1, 1)})
    with pytest.raises(ValueError):
        ContextTrace(CTX, ())


def test_total_capacity_constant_fits_the_v11_budget() -> None:
    # D1 §2: 512 items at roughly 430 bytes each stay under the 487,834-byte room.
    assert MAX_TRACE_ITEMS_TOTAL * 430 + MAX_TRACE_CONTEXTS * 300 < 487_834


def test_render_picks_by_retention_within_budget_then_orders_by_time() -> None:
    params = TraceParameters(max_rendered_bytes=300)
    old = _turn(1, T0, retention=0.3, user="old question " * 6, reply="old answer")
    new = _turn(2, T0 + timedelta(minutes=50), retention=0.95, user="new question", reply="new answer")
    huge = _turn(3, T0 + timedelta(minutes=40), retention=0.9, user="x" * 400, reply="y")
    rendered = render_trace([old, new, huge], T0 + timedelta(hours=1), params)
    assert rendered.text.startswith(RECENT_CONVERSATION_HEADING)
    assert rendered.shown_episode_ids == ("episode-0001", "episode-0002")
    assert rendered.faint_episode_ids == ("episode-0001",)
    lines = rendered.text.splitlines()
    assert lines[1].startswith("- about 1 h ago (faint) — User: ")
    assert "old question old question old question old question old ques… / You: old answer" in lines[1]
    assert lines[2] == "- about 10 min ago — User: new question / You: new answer"
    assert len(rendered.text.encode()) <= 300


def test_render_command_turns_as_a_fixed_note_without_ids() -> None:
    turn = _turn(1, T0, retention=0.9, user="REVIEW_EXPERIENCE:experience-x", reply="ok",
                 kind=TraceTurnKind.COMMAND)
    rendered = render_trace([turn], T0 + timedelta(minutes=5), TraceParameters())
    assert COMMAND_TURN_TEXT in rendered.text
    assert "experience" not in rendered.text


def test_render_nothing_when_empty() -> None:
    assert render_trace([], T0, TraceParameters()).text == ""


def test_settings_bounds_and_relations() -> None:
    settings = ConversationTraceSettings()
    assert settings.enabled is False
    assert TraceParameters.from_settings(settings) == TraceParameters()
    for bad in (
        {"forget_below": 0.6, "verbatim_above": 0.5},
        {"base_stability_hours": 0.0},
        {"base_stability_hours": 10.0, "max_stability_hours": 5.0},
        {"max_items_per_context": 33},
        {"recall_gain": -0.1},
    ):
        with pytest.raises(ValidationError):
            ConversationTraceSettings(**bad)  # type: ignore[arg-type]
    assert ConversationTraceSettings(base_stability_hours=0.05).base_stability_hours == 0.05


def test_command_matcher_is_shared_with_the_driver() -> None:
    target = "experience-00000000-0000-4000-8000-000000000001"
    assert parse_command(f"REVIEW_EXPERIENCE:{target}") == (R16CommandKind.REVIEW_EXPERIENCE, target)
    assert parse_command(f"COMMIT_REVIEW_EXPERIENCE:1:{target}") == (
        R16CommandKind.COMMIT_REVIEW_EXPERIENCE,
        target,
    )
    assert parse_command(f"please REVIEW_EXPERIENCE:{target}") is None
    assert driver_parse_command is parse_command


def _item_kwargs() -> dict[str, object]:
    return dict(
        episode_id="episode-0001",
        turn_kind=TraceTurnKind.CHAT,
        requested_at=T0,
        event_sequence=1,
        surprise=0.0,
        surprise_measured=True,
        stability_hours=1.0,
        last_reinforced_at=T0,
        last_reinforced_revision=1,
    )


def _turn(n: int, at: datetime, *, retention: float, user: str, reply: str,
          kind: TraceTurnKind = TraceTurnKind.CHAT) -> RecalledTurn:
    item = TraceItem(**{**_item_kwargs(), "episode_id": f"episode-{n:04d}", "requested_at": at,
                        "last_reinforced_at": at, "event_sequence": n, "turn_kind": kind})  # type: ignore[arg-type]
    return RecalledTurn(item=item, retention=retention, user_text=user, response_text=reply)
