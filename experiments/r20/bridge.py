"""The R15 S1 / S2 admission trigger (HD-22-2 (a); D1 fact 7, A-10).

No production code submits ``NARRATIVE_REVIEW``.  The harness pulls the
trigger: it submits the event to the app's own runtime and calls the app's own
``NarrativeSelfSystem`` owner (a private method, as every existing R15 test
does).  The admission itself is done by the real source gate and the real owner
inside a real runtime event; the manifest labels the path ``HARNESS_BRIDGE``.
Test-side use of a private name, behind the throwaway marker.
"""

from __future__ import annotations

from typing import Any

from suzka.memory.experience_participant import experience_id_for_event
from suzka.r15.qualified_proposal import QualifiedTurnRef
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import EventLifecycle

ADMISSION_PATH = "HARNESS_BRIDGE"
# The R15 protocol shape: setup, E, S1 request before E2; E2 and the S2 request
# after (``experiments/r15_p1``, #374 D3 sec. 5).
STAGE_TURNS = {"S1": 3, "S2": 5}
TIMEOUT_S = 30


class BridgeRefused(Exception):
    """A closed reason; the bridge never admits on a mismatch."""


def committed_chat_turns(state: Any) -> list[QualifiedTurnRef]:
    """The committed CHAT turns, in processing order."""

    refs: list[tuple[int, QualifiedTurnRef]] = []
    for record in state.event_journal.inspect().records:
        if record.lifecycle is not EventLifecycle.COMPLETED:
            continue
        if record.event_type is not AgentEventType.CHAT:
            continue
        entry = state.experience_store.load_committed_current(
            experience_id_for_event(record.event_id, record.processing_sequence)
        )
        refs.append(
            (
                record.processing_sequence,
                QualifiedTurnRef(record.event_id, record.processing_sequence, entry.record.source_episode_id),
            )
        )
    return [ref for _sequence, ref in sorted(refs, key=lambda item: item[0])]


def review(state: Any, stage: str, timeout_s: float = TIMEOUT_S) -> dict[str, object]:
    """Admit S1 (after exactly three committed turns) or S2 (after five)."""

    if stage not in STAGE_TURNS:
        raise BridgeRefused("STAGE_UNKNOWN")
    refs = committed_chat_turns(state)
    if len(refs) != STAGE_TURNS[stage]:
        raise BridgeRefused("TURN_COUNT_MISMATCH")
    owner = state.r15_state_port.narrative
    future = state.agent_runtime.submit(
        AgentEventType.NARRATIVE_REVIEW,
        AgentEventSource.RUNTIME_NARRATIVE_REVIEW,
        lambda: owner._review_committed_turns(tuple(refs)),
    )
    result = future.result(timeout=timeout_s).value
    return {
        "stage": stage,
        "changed": bool(result.changed),
        "revision": int(result.snapshot.revision),
        "path": ADMISSION_PATH,
    }


__all__ = ["ADMISSION_PATH", "STAGE_TURNS", "BridgeRefused", "committed_chat_turns", "review"]
