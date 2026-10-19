"""TEST-ONLY: read qualified U3 inside an *existing* serialized Chat event.

No production U6 consumer, prompt rewrite, Decision authority, WAL writes
from reading, or general subjective learning is implemented here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.private_consumer_preflight import (
    PrivateNarrativeProjection,
    PrivateReadStatus,
    read_private_committed_narrative,
)
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import EventLifecycle

from test_r15_u6_private_preflight import _scenario


def _qualified_view_in_current_chat(client: TestClient) -> PrivateNarrativeProjection:
    runtime = client.app.state.agent_runtime
    event = runtime.current_event()
    assert event is not None
    assert event.event_type is AgentEventType.CHAT
    assert event.source is AgentEventSource.API_CHAT
    assert type(event.processing_sequence) is int
    port = client.app.state.r15_state_port
    gate = port.narrative._source_gate
    assert gate is not None
    return read_private_committed_narrative(
        client.app.state.agent_state_store,
        port,
        gate,
    )


def _number_reviews(client: TestClient) -> int:
    return sum(
        item.lifecycle is EventLifecycle.COMPLETED
        and item.event_type is AgentEventType.NARRATIVE_REVIEW
        for item in client.app.state.event_journal.inspect().records
    )


@pytest.mark.parametrize("no_reveal", [False, True])
def test_current_private_view_is_readable_inside_existing_chat_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_reveal: bool,
) -> None:
    cases = _scenario(tmp_path, no_reveal=no_reveal)
    client, _source_refs, prior = next(cases)
    main_loop = client.app.state.main_loop
    original_prompt_builder = main_loop._build_prompt
    before_reviews = _number_reviews(client)
    before_root = client.app.state.r15_state_port.narrative.snapshot()
    observed: list[PrivateNarrativeProjection] = []

    def inspect_before_model_generation(*args, **kwargs) -> str:
        observed.append(_qualified_view_in_current_chat(client))
        # Do NOT inject interpretation or manipulate any model output.
        return original_prompt_builder(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(main_loop, "_build_prompt", inspect_before_model_generation)
        result = client.post(
            "/api/chat",
            json={
                "message": "合成の後続会話。モデルへ自己状態を直接設定しない。",
                "attachments": [],
            },
        )
    assert result.status_code == 200
    assert len(observed) == 1
    assert observed[0] == prior
    assert observed[0].status is PrivateReadStatus.QUALIFIED
    assert observed[0].current_referent == ("A" if no_reveal else "B")
    assert observed[0].original_referent == "A"
    assert not observed[0].human_intent_verified
    assert not observed[0].competence_verified
    assert _number_reviews(client) == before_reviews
    assert client.app.state.r15_state_port.narrative.snapshot() == before_root
    # Extra user Chat does not become another U3 subjective revision.
    list(cases)


def test_hold_during_chat_when_authoritative_wal_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _scenario(tmp_path, no_reveal=False)
    client, _refs, prior = next(cases)
    assert prior.status is PrivateReadStatus.QUALIFIED
    loop = client.app.state.main_loop
    original_prompt_builder = loop._build_prompt
    store = client.app.state.agent_state_store
    observed: list[PrivateNarrativeProjection] = []

    def probe_before_prompt(*args, **kwargs) -> str:
        # A deliberately injected WAL read failure affects only this
        # test-only read, not the subsequent real Chat internal commit.
        with monkeypatch.context() as patch:
            patch.setattr(store, "load", lambda: None)
            observed.append(_qualified_view_in_current_chat(client))
        return original_prompt_builder(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(loop, "_build_prompt", probe_before_prompt)
        response = client.post(
            "/api/chat",
            json={
                "message": "合成の次の会話。既存の自己仮説を勝手に断定しない。",
                "attachments": [],
            },
        )
    assert response.status_code == 200
    assert len(observed) == 1
    assert observed[0].status is PrivateReadStatus.HOLD
    assert observed[0].current_referent is None
    assert observed[0].original_referent is None
    assert client.app.state.r15_state_port.narrative.snapshot().revision == 2
    assert (
        client.app.state.agent_state_store.load().narrative_self_state["revision"] == 2
    )
    list(cases)
