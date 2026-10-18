"""P1b-B1: REAL isolated SUZKA StateWAL v10 owner persistence.

No production Source, no operator/model intrinsic setter. The provider is a
deterministic synthetic fixture; R07/R12/DB1/StateWAL and runtime are real.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.protocol import run_trial
from suzka.api.server import create_app
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.r15.qualified_proposal import restore_scoped_option
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.agent_state import (
    AgentStateSnapshotV9,
    AgentStateSnapshotV10,
)
from suzka.runtime.event_journal import EventLifecycle

from test_r15_p1_owner_bridge import scenario, SequencedProvider
from test_r15_p1b_positive import _admit, _source_ref


@pytest.mark.parametrize("no_reveal", [False, True])
def test_authoritative_r15_revision_persists_in_real_wal_and_restarts(
    tmp_path: Path,
    no_reveal: bool,
) -> None:
    app, provider, _source, manifest, capture = scenario(tmp_path, no_reveal=no_reveal)
    settings = app.state.settings
    references = []
    first_owner_review = None
    with TestClient(app) as client:
        runtime = client.app.state.agent_runtime
        owner = client.app.state.r15_state_port.narrative
        assert owner.snapshot().revision == 0

        def transport(message: str) -> dict[str, str]:
            nonlocal first_owner_review
            if len(references) == 3:
                assert capture.with_name(capture.name + ".s1.json").is_file()
                first_owner_review = _admit(runtime, owner, tuple(references))
                assert first_owner_review.changed
                assert first_owner_review.snapshot.revision == 1
                committed_first = client.app.state.agent_state_store.load()
                assert isinstance(committed_first, AgentStateSnapshotV10)
                assert committed_first.narrative_self_state["revision"] == 1
                assert committed_first.narrative_admission_checkpoint["stage"] == "S1"
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            data = response.json()
            references.append(_source_ref(client, data["episode_id"]))
            return {"episode_id": data["episode_id"], "response": data["response"]}

        run_trial(manifest, capture, transport)
        assert first_owner_review is not None
        if not no_reveal:
            second = _admit(runtime, owner, tuple(references))
            assert second.changed and second.snapshot.revision == 2
        exact_before_restart = owner.snapshot()
        loaded = client.app.state.agent_state_store.load()
        assert isinstance(loaded, AgentStateSnapshotV10)
        assert loaded.schema_version == 10
        assert loaded.narrative_self_state["revision"] == (1 if no_reveal else 2)
        assert loaded.narrative_admission_checkpoint["stage"] == (
            "S1" if no_reveal else "S2"
        )
        assert loaded.last_processed_event_sequence >= (
            exact_before_restart.revision_history[-1].event.event_sequence
        )
        assert exact_before_restart == (
            exact_before_restart.__class__.from_value(loaded.narrative_self_state)
        )
        assert loaded.relationship_state["revision"] == 0
        assert loaded.self_model_state["revision"] == 0
        journal = client.app.state.event_journal.inspect()
        reviews = [
            row
            for row in journal.records
            if row.event_type is AgentEventType.NARRATIVE_REVIEW
            and row.lifecycle is EventLifecycle.COMPLETED
        ]
        assert len(reviews) == (1 if no_reveal else 2)

    # Construct a completely fresh MainLoop, R15 owners and Journal lease.
    next_app = create_app(settings)
    next_app.state.model_provider = SequencedProvider(no_reveal=no_reveal)
    next_app.state.memory_system = DualMemorySystem(settings)
    next_app.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(next_app) as restarted:
        restored_owner = restarted.app.state.r15_state_port.narrative
        exact_restored = restored_owner.snapshot()
        assert exact_restored == exact_before_restart
        scope = first_owner_review.source_scope
        decoded = {
            restore_scoped_option(scope, ep.interpretation.statement_id)
            for ep in exact_restored.episodes
        }
        assert decoded == ({"A"} if no_reveal else {"A", "B"})
        replay = _admit(
            restarted.app.state.agent_runtime,
            restored_owner,
            tuple(references if not no_reveal else references[:3]),
        )
        assert replay.changed is False
        assert restored_owner.snapshot() == exact_restored
        final = restarted.app.state.agent_state_store.load()
        assert isinstance(final, AgentStateSnapshotV10)
        assert final.narrative_self_state["revision"] == (1 if no_reveal else 2)
        assert final.narrative_admission_checkpoint["source_scope"] == scope


def test_v9_stays_strict_and_has_no_claimed_r15_authority() -> None:
    assert "narrative_self_state" not in AgentStateSnapshotV9.model_fields
    assert "narrative_self_state" in AgentStateSnapshotV10.model_fields
    assert "narrative_admission_checkpoint" in AgentStateSnapshotV10.model_fields


def test_v10_checkpoint_cannot_be_rebound_to_arbitrary_subject_scope(
    tmp_path: Path,
) -> None:
    app, _provider, _source, _manifest, _capture = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        response = client.post(
            "/api/chat", json={"message": "a synthetic chat", "attachments": []}
        )
        assert response.status_code == 200
        current = client.app.state.agent_state_store.load()
        assert isinstance(current, AgentStateSnapshotV10)
        malicious = current.model_dump(mode="python")
        malicious["narrative_admission_checkpoint"]["source_scope"] = "forged"
        with pytest.raises((ValueError, TypeError)):
            AgentStateSnapshotV10.model_validate(malicious)
