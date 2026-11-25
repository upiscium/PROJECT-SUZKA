"""IP-22c, F14 (HD-22-5) as a TRANSITIONAL guard (#463).

The Human's final design lets a subject revise its interpretation, human-like
(tracked in issue #463).  Today NO such stage exists: a second R15
interpretation of the same X is refused (``narrative_self_system.py``,
``second reinterpretation needs a new stage`` / different original
experience).  That refusal is labelled ``TRANSITIONAL`` in the manifest and is
NOT asserted as final behaviour: the test records what happened and asserts
only what stays true under any later design:

1. committed S1 / S2 revisions are never silently rewritten and restore
   byte-identically after a restart;
2. a replay of the S2 admission with the same sources returns
   ``changed = False`` and changes nothing;
3. a changed interpretation is a NEW event (the later turns have their own event
   ids and later sequences) and never an in-place overwrite: whatever the admission
   outcome, revisions 1 and 2 and the Journal records of their review events
   are unchanged.

Not covered here (stated): the replay of a stored SEALED policy input against
its stored row for the new turn (the F14 (d) check) needs the R16 chain on an
R15 subject and belongs to the S0 dry run (22d).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.r15_p0.pilot import Case, seal_case
from experiments.r15_p1.protocol import run_trial
from experiments.r20 import bridge
from experiments.r20.factory import create_subject
from experiments.r20.manifest import Manifest
from experiments.r20.provider import ScriptedProvider, r15_answer, r15_protocol_script
from experiments.r20.tier1 import T1Runtime
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeExecutionError,
)
from suzka.runtime.agent_state import AgentStateSnapshotV10
from suzka.runtime.event_journal import EventLifecycle

TRANSITIONAL = "TRANSITIONAL"
COGNITIVE_VARIATION = "COGNITIVE_VARIATION"


def _admit(state: Any, refs: tuple[Any, ...]) -> Any:
    owner = state.r15_state_port.narrative
    future = state.agent_runtime.submit(
        AgentEventType.NARRATIVE_REVIEW,
        AgentEventSource.RUNTIME_NARRATIVE_REVIEW,
        lambda: owner._review_committed_turns(refs),
    )
    return future.result(timeout=bridge.TIMEOUT_S).value


def _checkpoint(state: Any) -> tuple[str, str]:
    loaded = state.agent_state_store.load()
    assert isinstance(loaded, AgentStateSnapshotV10)
    return (
        json.dumps(loaded.narrative_self_state, sort_keys=True),
        json.dumps(loaded.narrative_admission_checkpoint, sort_keys=True),
    )


def _review_events(state: Any) -> list[tuple[str, int, str]]:
    return [
        (row.event_id, row.processing_sequence, row.lifecycle.value)
        for row in state.event_journal.inspect().records
        if row.event_type is AgentEventType.NARRATIVE_REVIEW and row.lifecycle is EventLifecycle.COMPLETED
    ]


def _kept(before: Any, now: Any) -> bool:
    """Nothing committed was rewritten: the old revisions are a prefix."""

    return (
        now.revision >= before.revision
        and now.episodes[: len(before.episodes)] == before.episodes
        and now.revision_history[: len(before.revision_history)] == before.revision_history
    )


def test_f14_committed_revisions_are_never_rewritten_and_a_changed_interpretation_is_a_new_event(
    r20_base: Path,
) -> None:
    subject = create_subject("NEG-r15", base=r20_base)
    # the protocol answers, then a DIFFERENT later pair: the subject "changes its mind"
    later = (
        "a later supplement that points the other way",
        r15_answer("S2", "B"),
        r15_answer("S2", "A"),
    )
    provider = ScriptedProvider(r15_protocol_script() + later)
    manifest = Manifest(subject.evidence_dir, subject.run_id, subject_root=subject.root)
    seal = subject.evidence_dir / "p1.manifest.json"
    capture = subject.evidence_dir / "p1.capture.json"
    seal_case(Case("r20-22c-f14", "A", "forced", 0, "normal"), seal)
    runtime = T1Runtime(subject, provider)
    runtime.start()
    try:
        client, state = runtime.client, runtime.state
        assert client is not None

        def transport(message: str) -> dict[str, str]:
            if provider.calls == 3:  # after E and the S1 request, BEFORE E2
                bridge.review(state, "S1")
            reply = client.post("/api/chat", json={"message": message, "attachments": []})
            assert reply.status_code == 200
            data = reply.json()
            return {"episode_id": data["episode_id"], "response": data["response"]}

        run_trial(seal, capture, transport)
        assert bridge.review(state, "S2")["revision"] == 2
        committed = state.r15_state_port.narrative.snapshot()
        assert committed.revision == 2
        stored = (committed.canonical_bytes(), _checkpoint(state), _review_events(state))
        assert len(stored[2]) == 2
        refs = bridge.committed_chat_turns(state)
        assert len(refs) == 5
    finally:
        runtime.stop()

    # (1) restart: byte-identical restore of the committed revisions
    runtime.restart()
    try:
        client, state = runtime.client, runtime.state
        assert client is not None
        owner = state.r15_state_port.narrative
        assert owner.snapshot().canonical_bytes() == stored[0]
        assert _checkpoint(state) == stored[1]
        assert _review_events(state) == stored[2]

        # (2) replay of the S2 admission with the same sources: no change
        replay = bridge.review(state, "S2")
        assert replay["changed"] is False and replay["revision"] == 2
        assert owner.snapshot().canonical_bytes() == stored[0]
        assert _checkpoint(state) == stored[1]

        # (3) the subject's later turns answer DIFFERENTLY: a new attributed event
        turns_before = {ref.event_id for ref in bridge.committed_chat_turns(state)}
        for text in ("a later turn: on reflection I may have misread it", "and a second later turn"):
            reply = client.post("/api/chat", json={"message": text, "attachments": []})
            assert reply.status_code == 200
        grown = bridge.committed_chat_turns(state)
        fresh = [ref for ref in grown if ref.event_id not in turns_before]
        assert len(fresh) == 2
        assert all(ref.event_sequence > max(seq for _id, seq, _l in stored[2]) for ref in fresh)
        assert len({ref.event_id for ref in grown}) == len(grown)  # each turn its own event id

        # today's admission outcome for a second interpretation is OBSERVED, not pinned
        outcomes: list[str] = []
        for sources in (
            (*refs, *fresh),  # the old five plus the new turns
            (*refs[:3], *fresh),  # same original experience, other later sources
        ):
            try:
                result = _admit(state, tuple(sources))
                outcomes.append(f"ADMITTED:revision={result.snapshot.revision}")
            except AgentRuntimeExecutionError as error:
                cause = error.__cause__ or error.__context__
                outcomes.append(f"REFUSED:{type(cause).__name__ if cause else 'AgentRuntimeExecutionError'}")
            # whatever the outcome, nothing committed was rewritten in place
            assert _kept(committed, owner.snapshot())
        # the two committed review events are still exactly as they were
        assert [e for e in _review_events(state) if e in stored[2]] == stored[2]
        if all(item.startswith("REFUSED") for item in outcomes):
            # the state is then byte-identical (TRANSITIONAL: no revision stage exists yet)
            assert owner.snapshot().canonical_bytes() == stored[0]
            assert _checkpoint(state) == stored[1]
        # otherwise (a future revision stage): the old revisions stay a prefix (_kept above)

        manifest.result("F14.second_interpretation", TRANSITIONAL)
        manifest.result("F14.admission_outcome", ";".join(outcomes))
        manifest.result("F14.new_turn", COGNITIVE_VARIATION)
        manifest.append(
            "timeline",
            {"event_ids": [ref.event_id for ref in fresh], "kind": "chat", "label": COGNITIVE_VARIATION},
        )
        path = manifest.write()
    finally:
        runtime.stop()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["results"]["F14.second_interpretation"] == TRANSITIONAL
    assert written["results"]["F14.new_turn"] == COGNITIVE_VARIATION
    assert subject.root not in path.parents  # the manifest is outside the subject
