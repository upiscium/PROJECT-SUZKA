"""P1b-B1 source-qualified R15 WAL fault boundary tests (disposable only)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.protocol import run_trial
from suzka.api.server import create_app
from suzka.r15.continuity_state import ContinuityInvalid
from suzka.r15.qualified_proposal import QualifiedTurnRef, scoped_option_key
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


def test_private_checkpoint_cannot_rebind_witnessed_original_e(
    tmp_path: Path,
) -> None:
    """An ordered but forged E reference must not survive U3 restore."""
    with source_ready_for_later_reinterpretation(tmp_path) as (
        client,
        runtime,
        owner,
        refs,
        _settings,
    ):
        root, history = owner._technical_state_for_agent_state()
        assert root.revision == 1
        original = history.first_sources[1]
        forged = QualifiedTurnRef(
            "forged-original-e-event", original.event_sequence, original.episode_id
        )
        changed = replace(
            history,
            first_sources=(history.first_sources[0], forged, history.first_sources[2]),
        )
        with pytest.raises(ContinuityInvalid, match="original E reference"):
            changed.canonical_value(root)
        stored = client.app.state.agent_state_store.load()
        assert isinstance(stored, AgentStateSnapshotV10)
        tampered = stored.model_dump(mode="python")
        tampered["narrative_admission_checkpoint"]["first_sources"][1]["event_id"] = (
            forged.event_id
        )
        with pytest.raises(ValueError):
            AgentStateSnapshotV10.model_validate(tampered)
        assert owner.snapshot() == root


def test_private_checkpoint_cannot_rebind_witnessed_later_e2(
    tmp_path: Path,
) -> None:
    with source_ready_for_later_reinterpretation(tmp_path) as (
        client,
        runtime,
        owner,
        refs,
        _settings,
    ):
        admitted = _admit(runtime, owner, refs)
        assert admitted.changed and admitted.snapshot.revision == 2
        root, history = owner._technical_state_for_agent_state()
        original = history.revised_sources[3]
        forged = QualifiedTurnRef(
            "forged-later-e2-event", original.event_sequence, original.episode_id
        )
        revised = list(history.revised_sources)
        revised[3] = forged
        changed = replace(history, revised_sources=tuple(revised))
        with pytest.raises(ContinuityInvalid, match="reported E2"):
            changed.canonical_value(root)
        stored = client.app.state.agent_state_store.load()
        assert isinstance(stored, AgentStateSnapshotV10)
        tampered = stored.model_dump(mode="python")
        tampered["narrative_admission_checkpoint"]["revised_sources"][3]["event_id"] = (
            forged.event_id
        )
        with pytest.raises(ValueError):
            AgentStateSnapshotV10.model_validate(tampered)
        assert owner.snapshot() == root


def test_unknown_referent_is_not_admitted_as_source_qualified_positive(
    tmp_path: Path,
) -> None:
    """UNKNOWN remains a held/unadmitted hypothesis in this finite MVI."""
    with source_ready_for_later_reinterpretation(tmp_path) as (
        _client,
        _runtime,
        owner,
        _refs,
        _settings,
    ):
        root, history = owner._technical_state_for_agent_state()
        assert root.revision == 1
        original = root.episodes[0]
        from suzka.r15.contracts import NarrativeSelfState
        from suzka.r15.qualified_proposal import restore_scoped_option

        unknown_interpretation = replace(
            original.interpretation,
            statement_id=scoped_option_key(history.source_scope, "UNKNOWN"),
        )
        assert (
            restore_scoped_option(
                history.source_scope, unknown_interpretation.statement_id
            )
            == "UNKNOWN"
        )
        hypothetical = NarrativeSelfState(
            1,
            1,
            (replace(original, interpretation=unknown_interpretation),),
            root.claims,
            root.history_anchor,
            root.revision_history,
        )
        with pytest.raises(ContinuityInvalid, match="UNKNOWN"):
            replace(
                history,
                root_digest=hypothetical.canonical_value()["state_digest"],
            ).canonical_value(hypothetical)
        assert owner.snapshot() == root
