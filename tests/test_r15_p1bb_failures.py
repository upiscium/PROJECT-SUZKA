"""P1b-B1 source-qualified R15 WAL fault boundary tests (disposable only)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.protocol import run_trial
from suzka.api.server import create_app
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeDurabilityError,
    AgentRuntimeDurabilityPhase,
    AgentRuntimeStatus,
)
from suzka.runtime.agent_state import AgentStateSnapshotV10

from test_r15_p1_owner_bridge import scenario, SequencedProvider
from test_r15_p1b_positive import _admit, _source_ref


@contextmanager
def source_ready_for_later_reinterpretation(tmp_path: Path) -> Iterator[tuple]:
    app, _provider, _source, manifest, capture = scenario(tmp_path)
    settings = app.state.settings
    with TestClient(app) as client:
        runtime = client.app.state.agent_runtime
        owner = client.app.state.r15_state_port.narrative
        refs = []
        first = None

        def transport(message: str) -> dict[str, str]:
            nonlocal first
            if len(refs) == 3:
                assert capture.with_name(capture.name + ".s1.json").exists()
                first = _admit(runtime, owner, tuple(refs))
                assert first.changed and first.snapshot.revision == 1
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            data = response.json()
            refs.append(_source_ref(client, data["episode_id"]))
            return {"episode_id": data["episode_id"], "response": data["response"]}

        run_trial(manifest, capture, transport)
        assert first is not None
        assert owner.snapshot().revision == 1
        loaded = client.app.state.agent_state_store.load()
        assert isinstance(loaded, AgentStateSnapshotV10)
        assert loaded.narrative_self_state["revision"] == 1
        yield client, runtime, owner, tuple(refs), settings


def test_capture_pre_wal_fault_rolls_back_owner_to_last_committed_s1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with source_ready_for_later_reinterpretation(tmp_path) as (
        client,
        runtime,
        owner,
        refs,
        _settings,
    ):
        canonical_before = owner.snapshot().canonical_bytes()
        store = client.app.state.agent_state_store

        def fail_capture(*args, **kwargs):
            # Exactly one following review; handler event identity is
            # intentionally unavailable at internal-commit time.
            raise RuntimeError("synthetic failure before WAL candidate")

        with monkeypatch.context() as context:
            context.setattr(store, "capture", fail_capture)
            future = runtime.submit(
                AgentEventType.NARRATIVE_REVIEW,
                AgentEventSource.RUNTIME_NARRATIVE_REVIEW,
                lambda: owner._review_committed_turns(refs),
            )
            with pytest.raises(AgentRuntimeDurabilityError) as failure:
                future.result(timeout=60)
        assert failure.value.phase is AgentRuntimeDurabilityPhase.INTERNAL_COMMIT
        assert runtime.status is AgentRuntimeStatus.FAILED
        assert owner.snapshot().canonical_bytes() == canonical_before
        after = store.load()
        assert isinstance(after, AgentStateSnapshotV10)
        assert after.narrative_self_state["revision"] == 1
        assert after.narrative_admission_checkpoint["stage"] == "S1"


def test_wal_committed_but_not_terminal_event_recovers_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with source_ready_for_later_reinterpretation(tmp_path) as (
        client,
        runtime,
        owner,
        refs,
        settings,
    ):
        recovery = client.app.state.state_recovery
        complete = recovery.complete_committed_event

        def fail_before_terminal(event, evidence):
            if event.event_type is AgentEventType.NARRATIVE_REVIEW:
                raise RuntimeError("simulate process loss after WAL publication")
            return complete(event, evidence)

        with monkeypatch.context() as context:
            context.setattr(recovery, "complete_committed_event", fail_before_terminal)
            future = runtime.submit(
                AgentEventType.NARRATIVE_REVIEW,
                AgentEventSource.RUNTIME_NARRATIVE_REVIEW,
                lambda: owner._review_committed_turns(refs),
            )
            with pytest.raises(AgentRuntimeDurabilityError) as failure:
                future.result(timeout=60)
        assert failure.value.phase is AgentRuntimeDurabilityPhase.TERMINAL_COMPLETION
        assert runtime.status is AgentRuntimeStatus.FAILED
        assert owner.snapshot().revision == 2
        durable = client.app.state.agent_state_store.load()
        assert isinstance(durable, AgentStateSnapshotV10)
        assert durable.narrative_self_state["revision"] == 2
        # The important point: DO NOT roll back or resubmit an unknown WAL
        # transition just because R07 completion was interrupted.

    restarted_app = create_app(settings)
    restarted_app.state.model_provider = SequencedProvider()
    restarted_app.state.memory_system = DualMemorySystem(settings)
    restarted_app.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(restarted_app) as restarted:
        actual = restarted.app.state.r15_state_port.narrative
        assert actual.snapshot().revision == 2
        saved = restarted.app.state.agent_state_store.load()
        assert isinstance(saved, AgentStateSnapshotV10)
        assert saved.narrative_self_state["revision"] == 2
        # If startup recovered to an accepting state, a new review of the
        # already recorded evidence cannot add another owned revision.
        if restarted.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING:
            outcome = _admit(restarted.app.state.agent_runtime, actual, refs)
            assert outcome.changed is False
            assert actual.snapshot().revision == 2
        else:
            assert restarted.app.state.external_reconciliation_required
