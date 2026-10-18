"""R15 U6 private-read PRE-FLIGHT, not a production Decision/prompt API.

The deterministic chat provider is a SYNTHETIC model fixture. All R07/R12,
DB1, AgentState v10, StateWAL and source checks use real isolated owners.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.private_consumer_preflight import (
    PrivateReadStatus,
    read_private_committed_narrative,
)
from experiments.r15_p1.protocol import run_trial
from suzka.api.server import create_app
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.memory.experience_participant import experience_id_for_event
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.agent_state import AgentStateSnapshotV10
from suzka.r15.state_port import R15TechnicalSnapshot

from test_r15_p1_owner_bridge import SequencedProvider, scenario
from test_r15_p1b_positive import _admit, _source_ref


def _view(client: TestClient):
    port = client.app.state.r15_state_port
    gate = port.narrative._source_gate
    assert gate is not None
    return read_private_committed_narrative(
        client.app.state.agent_state_store,
        port,
        gate,
    )


def _scenario(tmp_path: Path, *, no_reveal: bool):
    app, _provider, _source, manifest, capture = scenario(tmp_path, no_reveal=no_reveal)
    settings = app.state.settings
    refs = []
    with TestClient(app) as client:
        runtime = client.app.state.agent_runtime
        owner = client.app.state.r15_state_port.narrative
        assert _view(client).status is PrivateReadStatus.HOLD

        def transport(msg: str) -> dict[str, str]:
            if len(refs) == 3:
                assert capture.with_name(capture.name + ".s1.json").exists()
                initial = _admit(runtime, owner, tuple(refs))
                assert initial.changed and initial.snapshot.revision == 1
                private = _view(client)
                assert private.status is PrivateReadStatus.QUALIFIED
                assert private.current_referent == "A"
                assert private.original_referent == "A"
                assert private.revision == 1
            resp = client.post("/api/chat", json={"message": msg, "attachments": []})
            assert resp.status_code == 200
            body = resp.json()
            refs.append(_source_ref(client, body["episode_id"]))
            return {"episode_id": body["episode_id"], "response": body["response"]}

        run_trial(manifest, capture, transport)
        if not no_reveal:
            updated = _admit(runtime, owner, tuple(refs))
            assert updated.changed and updated.snapshot.revision == 2
        private = _view(client)
        assert private.status is PrivateReadStatus.QUALIFIED
        assert private.revision == (1 if no_reveal else 2)
        assert private.current_referent == ("A" if no_reveal else "B")
        assert private.original_referent == "A"
        assert private.later_user_reported_referent == (not no_reveal)
        assert private.human_intent_verified is False
        assert private.competence_verified is False
        assert {item.name for item in fields(private)} <= {
            "status",
            "revision",
            "current_referent",
            "original_referent",
            "later_user_reported_referent",
            "human_intent_verified",
            "competence_verified",
        }
        assert client.app.state.agent_state_store.load().schema_version == 10
        yield client, tuple(refs), private

    app2 = create_app(settings)
    app2.state.model_provider = SequencedProvider(no_reveal=no_reveal)
    app2.state.memory_system = DualMemorySystem(settings)
    app2.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(app2) as restarted:
        current = _view(restarted)
        assert current == private
        assert isinstance(
            restarted.app.state.agent_state_store.load(), AgentStateSnapshotV10
        )
        before_root = restarted.app.state.r15_state_port.narrative.snapshot()
        before_events = tuple(
            record
            for record in restarted.app.state.event_journal.inspect().records
            if record.event_type is AgentEventType.NARRATIVE_REVIEW
        )
        for _ in range(2):
            assert _view(restarted) == private
        assert restarted.app.state.r15_state_port.narrative.snapshot() == before_root
        after_events = tuple(
            record
            for record in restarted.app.state.event_journal.inspect().records
            if record.event_type is AgentEventType.NARRATIVE_REVIEW
        )
        assert after_events == before_events
        yield restarted, tuple(refs), current


@pytest.mark.parametrize("no_reveal", [False, True])
def test_first_person_current_view_is_recovered_without_new_adoption(
    tmp_path: Path, no_reveal: bool
) -> None:
    stages = list(_scenario(tmp_path, no_reveal=no_reveal))
    assert len(stages) == 2
    assert stages[0][2] == stages[1][2]


@pytest.mark.parametrize("index", [1, 3])
def test_missing_original_or_later_experience_revokes_private_view(
    tmp_path: Path,
    index: int,
) -> None:
    cases = _scenario(tmp_path, no_reveal=False)
    client, refs, view = next(cases)
    owner = client.app.state.r15_state_port.narrative
    before = owner.snapshot()
    first_e = refs[index]
    store = client.app.state.experience_store
    eid = experience_id_for_event(first_e.event_id, first_e.event_sequence)
    path = store.record_path(eid, 0)
    assert path.is_file()
    saved = path.read_bytes()
    path.unlink()
    try:
        held = _view(client)
        assert held.status is PrivateReadStatus.HOLD
        assert held.current_referent is None
        assert held.revision is None
        assert owner.snapshot() == before
    finally:
        path.write_bytes(saved)
        path.chmod(0o600)
    assert _view(client) == view
    list(cases)


def test_unknown_person_identity_does_not_convert_into_qualified_relation(
    tmp_path: Path,
) -> None:
    cases = _scenario(tmp_path, no_reveal=True)
    client, _refs, view = next(cases)
    registry = client.app.state.main_loop.context_registry
    context_id = registry.current_context_id
    assert context_id is not None
    registry.add_participant_ref(context_id, "unverified-new-person")
    hidden = _view(client)
    assert hidden.status is PrivateReadStatus.HOLD
    assert hidden.current_referent is None
    # E2 missing is not a positive relation/person proof.
    assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
    # The synthetic Context is deliberately changed; do not ask this
    # consumed fixture to assert that its persisted identity is unchanged.
    cases.close()


def test_no_public_model_or_arbitrary_snapshot_is_consumer_authority() -> None:
    with pytest.raises(TypeError):
        read_private_committed_narrative(
            object(),
            object(),
            object(),  # type: ignore[arg-type]
        )


def test_uncommitted_process_local_root_is_not_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _scenario(tmp_path, no_reveal=False)
    client, _refs, selected = next(cases)
    port = client.app.state.r15_state_port
    store = client.app.state.agent_state_store
    original_bytes = store.load().narrative_self_state
    with monkeypatch.context() as patch:
        patch.setattr(
            port,
            "_capture_for_agent_state",
            lambda: R15TechnicalSnapshot.empty(),
        )
        held = _view(client)
        assert held.status is PrivateReadStatus.HOLD
        assert held.current_referent is None
        assert held.revision is None
    assert store.load().narrative_self_state == original_bytes
    assert _view(client) == selected
    list(cases)


def test_unavailable_wal_snapshot_is_not_a_self_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _scenario(tmp_path, no_reveal=True)
    client, _refs, selected = next(cases)
    store = client.app.state.agent_state_store
    with monkeypatch.context() as patch:
        patch.setattr(store, "load", lambda: None)
        held = _view(client)
        assert held.status is PrivateReadStatus.HOLD
        assert held.current_referent is None
        assert held.revision is None
    assert _view(client) == selected
    list(cases)
