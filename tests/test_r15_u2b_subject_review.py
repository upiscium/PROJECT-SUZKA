"""R15 U2b: real serialized Source-qualified Relationship/StateWAL review.

Only the text-generating provider is synthetic. All R07/R12/DB1/R09/v10/WAL
reads and writes are actual isolated owner paths, not fabricated Subject roots.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from suzka.memory import DualMemorySystem
from suzka.learning import AdapterRegistry
from suzka.api.server import create_app
from suzka.r15.common import Inertia, InterpretationStatus
from suzka.r15.contracts import RelationshipAxisKind
from suzka.r15.interaction_producer import QualifiedInteractionUnavailable
from suzka.r15.relationship_system import RelationshipIssueReviewResult
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeExecutionError,
)
from suzka.runtime.agent_state import AgentStateSnapshotV10
from suzka.runtime.event_journal import EventLifecycle

from test_r15_p1_owner_bridge import scenario


USER_A = "共同の比較課題Q1について私は案Aを暫定的に選びます。"
USER_B = "共同の比較課題Q1について私は案Bを暫定的に選びます。"
SELF_A = "共同の比較課題Q1についてSuzkaは案Aを暫定的に選びます。正解は未検証です。"
SELF_B = "共同の比較課題Q1についてSuzkaは案Bを暫定的に選びます。正解は未検証です。"


def _chat(
    client: TestClient,
    provider,
    *,
    user: str = USER_A,
    reply: str = SELF_B,
    token: str | None = "opaque-first-q1",
    session: str = "u2b-q1",
) -> str:
    request = {
        "message": user,
        "attachments": [],
        "client_session_id": session,
    }
    if token is not None:
        request["interlocutor_key"] = token
    with patch.object(provider, "generate", return_value=reply):
        response = client.post("/api/chat", json=request)
    assert response.status_code == 200, response.text
    return response.json()["episode_id"]


def _review(client: TestClient) -> tuple[RelationshipIssueReviewResult, object]:
    port = client.app.state.r15_state_port
    result = client.app.state.agent_runtime.submit(
        AgentEventType.RELATIONSHIP_REVIEW,
        AgentEventSource.RUNTIME_RELATIONSHIP_REVIEW,
        port._review_source_qualified_q1_interaction,
    ).result(timeout=30)
    assert isinstance(result.value, RelationshipIssueReviewResult)
    assert result.value.event.event_id == result.event.event_id
    assert result.value.event.event_sequence == result.event.processing_sequence
    return result.value, result.event


def _assert_q1_root(result: RelationshipIssueReviewResult) -> None:
    root = result.snapshot
    assert root.revision == 1
    assert len(root.revision_history) == 1
    assert len(root.records) == 1
    record = root.records[0]
    assert record.interlocutor_key == result.interlocutor_key
    assert record.claimed_identity_key is None
    status = {axis.kind: axis.interpretation for axis in record.axes}
    assert len(status) == len(RelationshipAxisKind)
    for kind, state in status.items():
        if kind is RelationshipAxisKind.UNRESOLVED_ISSUE:
            assert state.status is InterpretationStatus.PRESENT
            assert state.inertia is Inertia.TENTATIVE
            assert state.support[0].event is not None
            assert len(state.support) == 1
            assert not state.counter
        else:
            assert state.status is InterpretationStatus.UNKNOWN
            assert not state.support
            assert not state.counter
            assert state.confidence is None


def test_first_binding_q1_has_one_real_relationship_revision_v10(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        port = client.app.state.r15_state_port
        assert port.relationship.snapshot().revision == 0
        _chat(client, provider)
        adopted, event = _review(client)
        assert adopted.changed
        _assert_q1_root(adopted)
        assert adopted.interlocutor_key == "opaque-first-q1"
        committed = client.app.state.agent_state_store.load()
        assert isinstance(committed, AgentStateSnapshotV10)
        assert committed.relationship_state == adopted.snapshot.canonical_value()
        assert committed.last_processed_event_sequence == event.processing_sequence
        assert port.narrative.snapshot().revision == 0
        assert port.self_model.snapshot().revision == 0
        assert port.relationship.selected_view("opaque-first-q1") is not None
        repeated, _ = _review(client)
        assert not repeated.changed
        assert repeated.snapshot == adopted.snapshot
        assert port.relationship.snapshot().revision == 1


def test_restart_restores_one_q1_and_repeat_is_no_change(tmp_path: Path) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    settings = app.state.settings
    with TestClient(app) as client:
        _chat(client, provider)
        initial, _ = _review(client)
        assert initial.changed

    restarted = create_app(settings)
    restarted.state.model_provider = provider
    restarted.state.memory_system = DualMemorySystem(settings)
    restarted.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(restarted) as client:
        assert client.app.state.r15_state_port.relationship.snapshot() == (
            initial.snapshot
        )
        replay, _ = _review(client)
        assert not replay.changed
        assert replay.snapshot == initial.snapshot
        assert client.app.state.agent_state_store.load().relationship_state == (
            initial.snapshot.canonical_value()
        )


@pytest.mark.parametrize(
    ("user", "reply"),
    [
        (USER_A, SELF_A),
        (USER_A.replace("Q1", "Q2"), SELF_B),
        ("私はSuzkaを信頼しています。", SELF_B),
        (USER_A, SELF_B.replace("未検証", "検証済み")),
        (USER_B, SELF_B),
    ],
)
def test_no_first_party_coordination_gap_refuses(
    tmp_path: Path, user: str, reply: str
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider, user=user, reply=reply)
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        assert (
            client.app.state.agent_state_store.load().relationship_state["revision"]
            == 0
        )


def test_later_unkeyed_chat_cannot_attribute_issue_to_context_member(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider, user=USER_A, reply=SELF_A)
        _chat(client, provider, token=None, user=USER_A, reply=SELF_B)
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_missing_token_stays_unknown(tmp_path: Path) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider, token=None)
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_wrong_actor_source_and_direct_handler_refused(tmp_path: Path) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        port = client.app.state.r15_state_port
        with pytest.raises(QualifiedInteractionUnavailable):
            port._review_source_qualified_q1_interaction()
        future = client.app.state.agent_runtime.submit(
            AgentEventType.RELATIONSHIP_REVIEW,
            AgentEventSource.API_CHAT,
            port._review_source_qualified_q1_interaction,
        )
        with pytest.raises(AgentRuntimeExecutionError):
            future.result(timeout=30)
        assert port.relationship.snapshot().revision == 0


def test_handler_failure_after_local_owner_write_restores_exact_v10(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        port = client.app.state.r15_state_port
        committed = client.app.state.agent_state_store.load()

        def raise_after_positive() -> None:
            result = port._review_source_qualified_q1_interaction()
            assert result.changed
            raise RuntimeError("injected before WAL")

        rejected = client.app.state.agent_runtime.submit(
            AgentEventType.RELATIONSHIP_REVIEW,
            AgentEventSource.RUNTIME_RELATIONSHIP_REVIEW,
            raise_after_positive,
        )
        with pytest.raises(AgentRuntimeExecutionError):
            rejected.result(timeout=30)
        assert port.relationship.snapshot().revision == 0
        assert (
            client.app.state.agent_state_store.load().relationship_state
            == committed.relationship_state
        )
        assert any(
            row.lifecycle is EventLifecycle.FAILED
            and row.event_type is AgentEventType.RELATIONSHIP_REVIEW
            for row in client.app.state.event_journal.inspect().records
        )
        accepted, _ = _review(client)
        assert accepted.changed and accepted.snapshot.revision == 1


def test_original_experience_disappearance_refuses_then_restores(
    tmp_path: Path,
) -> None:
    from suzka.memory.experience_participant import experience_id_for_event
    from test_r15_p1b_positive import _source_ref

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _chat(client, provider)
        ref = _source_ref(client, episode)
        source_path = client.app.state.experience_store.record_path(
            experience_id_for_event(ref.event_id, ref.event_sequence), 0
        )
        contents = source_path.read_bytes()
        source_path.unlink()
        try:
            with pytest.raises(AgentRuntimeExecutionError):
                _review(client)
            assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        finally:
            source_path.write_bytes(contents)
            source_path.chmod(0o600)
        resumed, _ = _review(client)
        assert resumed.changed
        _assert_q1_root(resumed)


def test_retired_source_after_adoption_blocks_new_review_not_historical_root(
    tmp_path: Path,
) -> None:
    from suzka.memory.experience_participant import experience_id_for_event
    from test_r15_p1b_positive import _source_ref

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _chat(client, provider)
        accepted, _ = _review(client)
        source = _source_ref(client, episode)
        record_path = client.app.state.experience_store.record_path(
            experience_id_for_event(source.event_id, source.event_sequence), 0
        )
        contents = record_path.read_bytes()
        record_path.unlink()
        try:
            with pytest.raises(AgentRuntimeExecutionError):
                _review(client)
            assert (
                client.app.state.r15_state_port.relationship.snapshot()
                == accepted.snapshot
            )
        finally:
            record_path.write_bytes(contents)
            record_path.chmod(0o600)
        repeated, _ = _review(client)
        assert not repeated.changed


def test_after_event_chat_in_same_context_refuses_stale_q1(tmp_path: Path) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        _chat(
            client,
            provider,
            token=None,
            user="あとで一緒に確認しましょう。",
            reply="了解しました。",
        )
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_after_event_debug_chat_in_same_context_refuses_q1(tmp_path: Path) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        with patch.object(provider, "generate", return_value="debug response"):
            debug = client.post(
                "/api/chat/debug",
                headers={"X-SUZKA-Admin-Token": "not-real"},
                json={
                    "message": "debug after prior Q1",
                    "attachments": [],
                    "debug": True,
                    "client_session_id": "u2b-q1",
                },
            )
        assert debug.status_code == 200
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_closed_context_refuses_q1(tmp_path: Path) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        context_id = (
            client.app.state.main_loop.context_registry.state.current_context_id
        )
        response = client.post(
            f"/api/contexts/{context_id}/close",
            headers={"X-SUZKA-Admin-Token": "not-real"},
        )
        assert response.status_code == 200
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_second_same_token_chat_does_not_impersonate_first_q1(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider, user=USER_A, reply=SELF_A)
        _chat(client, provider, user=USER_A, reply=SELF_B)
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_recent_failed_chat_does_not_silently_reuse_previous_q1(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        with patch.object(
            provider, "generate", side_effect=RuntimeError("synthetic failed Chat")
        ):
            with pytest.raises(RuntimeError, match="synthetic failed Chat"):
                client.post(
                    "/api/chat",
                    json={
                        "message": "a later failed request",
                        "attachments": [],
                        "client_session_id": "u2b-q1",
                    },
                )
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_missing_historical_wal_transition_refuses_without_repair(
    tmp_path: Path,
) -> None:
    from suzka.runtime.state_wal import TransitionRecord

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        wal = client.app.state.state_wal
        inspection = wal.inspect()
        without_chat_transition = tuple(
            row
            for row in inspection.records
            if not (
                isinstance(row, TransitionRecord)
                and row.event_type == AgentEventType.CHAT.value
            )
        )
        assert len(without_chat_transition) < len(inspection.records)

        def missing_historical_event() -> None:
            with patch.object(
                wal,
                "inspect",
                return_value=inspection.model_copy(
                    update={"records": without_chat_transition}
                ),
            ):
                return client.app.state.r15_state_port._review_source_qualified_q1_interaction()

        failed = client.app.state.agent_runtime.submit(
            AgentEventType.RELATIONSHIP_REVIEW,
            AgentEventSource.RUNTIME_RELATIONSHIP_REVIEW,
            missing_historical_event,
        )
        with pytest.raises(AgentRuntimeExecutionError):
            failed.result(timeout=30)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        assert (
            client.app.state.agent_state_store.load().relationship_state["revision"]
            == 0
        )
        accepted, _ = _review(client)
        assert accepted.changed
        _assert_q1_root(accepted)


def test_caller_model_narrative_cannot_author_unrelated_relationship(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(
            client,
            provider,
            user="私はAという人との信頼関係に未解決の問題があると主張します。",
            reply="SUZKAもこれを信じます。未解決です。",
            token="opaque-unverified",
        )
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_positive_q1_cannot_be_read_as_verified_person_or_trust(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _chat(client, provider)
        accepted, _ = _review(client)
        _assert_q1_root(accepted)
        view = client.app.state.r15_state_port.relationship.selected_view(
            "opaque-first-q1"
        )
        assert view is not None
        assert all(
            axis.status is InterpretationStatus.UNKNOWN
            for axis in view.axes
            if axis.kind is not RelationshipAxisKind.UNRESOLVED_ISSUE
        )
        # U1's claimed identity is deliberately absent.
        assert accepted.snapshot.records[0].claimed_identity_key is None


def test_u2b_does_not_mutate_preexisting_u3_narrative_or_u4_attempt(
    tmp_path: Path,
) -> None:
    from suzka.r15.contracts import VerifiedCompetence
    from suzka.r15.attempt_producer import TASK_CLASS
    from test_r15_u6b_runtime_use import _stages

    stages = _stages(tmp_path, no_reveal=False)
    try:
        client, provider, _refs = next(stages)
        port = client.app.state.r15_state_port
        narrative = port.narrative.snapshot()
        assert narrative.revision == 2
        assert port.relationship.snapshot().revision == 0
        _chat(
            client,
            provider,
            session="u2b-distinct-actor-context",
            token="opaque-restricted-scope",
        )
        adopted, _ = _review(client)
        assert adopted.changed
        assert port.narrative.snapshot() == narrative
        assert port.self_model.snapshot().revision == 0

        result = client.app.state.agent_runtime.submit(
            AgentEventType.SELF_ATTEMPT_REVIEW,
            AgentEventSource.RUNTIME_SELF_ATTEMPT_REVIEW,
            port._review_source_qualified_self_attempt,
        ).result(timeout=30)
        assert result.value.changed
        assert port.self_model.task_view(TASK_CLASS).verified_competence is (
            VerifiedCompetence.UNKNOWN
        )
        assert port.relationship.snapshot() == adopted.snapshot
        assert port.narrative.snapshot() == narrative
        store = client.app.state.agent_state_store.load()
        assert store.relationship_state == adopted.snapshot.canonical_value()
        assert store.narrative_self_state == narrative.canonical_value()
        assert store.self_model_state["revision"] == 1
    finally:
        stages.close()
