"""P1b-B1 source-qualified R15 WAL fault boundary tests (disposable only)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from inspect import getfile
import json
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.protocol import run_trial
from suzka.api.server import create_app
from suzka.r15.continuity_state import ContinuityInvalid
from suzka.r15.common import canonical_json
from suzka.r15.narrative_self_system import NarrativeSelfSystem
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
    scoped_option_key,
)
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.memory.experience_participant import experience_id_for_event
from suzka.persona.prompt_builder import PromptBuilder
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    AgentRuntimeDurabilityError,
    AgentRuntimeDurabilityPhase,
    AgentRuntimeExecutionError,
    AgentRuntimeStatus,
)
from suzka.runtime.agent_state import AgentStateSnapshotV10, AgentStateStore
from suzka.runtime.event_journal import EventLifecycle, EventRecoveryCategory
from suzka.runtime.startup_reconciliation import (
    StartupReconciliationCoordinator,
    StartupReconciliationResult,
)
from suzka.runtime.state_recovery import (
    StateRecoveryCoordinator,
    StateRecoveryError,
    StateRecoveryResult,
)
from suzka.runtime.state_wal import StateWAL, StateWALIntegrityError
from suzka.runtime.working_memory import WorkingMemory

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


def test_true_rollback_restores_retained_nonempty_r15_v10_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On-disk S2 loss selects exact S1, not cognition or new source authority."""
    repo = Path(__file__).resolve().parents[1]
    for implementation in (
        StateRecoveryCoordinator,
        AgentStateStore,
        StateWAL,
        StartupReconciliationCoordinator,
        NarrativeSelfSystem,
    ):
        assert Path(getfile(implementation)).resolve().is_relative_to(repo / "suzka")

    with source_ready_for_later_reinterpretation(tmp_path) as (
        client,
        runtime,
        owner,
        refs,
        settings,
    ):
        store = client.app.state.agent_state_store
        journal = client.app.state.event_journal
        wal = client.app.state.state_wal
        recovery = client.app.state.state_recovery
        older = store.load()
        assert isinstance(older, AgentStateSnapshotV10)
        technical = client.app.state.r15_state_port._capture_for_agent_state()
        narrative_bytes = technical.narrative.canonical_bytes()
        continuation_bytes = canonical_json(
            technical.continuation.canonical_value(technical.narrative)
        )
        assert older.narrative_self_state == technical.narrative.canonical_value()
        assert older.narrative_self_state["revision"] == 1
        assert older.narrative_self_state["episodes"]
        assert older.narrative_admission_checkpoint["stage"] == "S1"
        assert technical.continuation.first_sources == refs[:3]
        assert technical.continuation.revised_sources == ()
        assert older.relationship_state["revision"] == 0
        assert older.self_model_state["revision"] == 0
        older_bytes = store.path.read_bytes()
        anchored = recovery.prepare_startup()
        recovery.publish_boot_anchor(anchored)
        anchor = wal.inspect_boot_anchor()
        predecessor = wal.root / "generations" / f"{anchor.generation_id}.jsonl"
        predecessor_bytes = predecessor.read_bytes()
        assert wal.inspect_anchored_prefix()[1] == older

        # A torn manifest with valid canonical S1 invokes real exact-current
        # repair, creating a distinct successor without touching the retained
        # generation or its boot anchor. No fabricated recovery category/setter.
        (wal.root / "manifest.json").write_bytes(b"{")
        repaired = recovery.prepare_startup()
        assert repaired.exact_current_reconstructed
        assert not repaired.true_rollback_performed
        assert not repaired.external_reconciliation_required
        assert repaired.snapshot == older
        successor = repaired.manifest
        assert successor.active_generation_id != anchor.generation_id
        assert recovery.prepare_startup().snapshot == older
        second = _admit(runtime, owner, refs)
        assert second.changed and second.snapshot.revision == 2
        newer = store.load()
        assert isinstance(newer, AgentStateSnapshotV10)
        assert newer.narrative_self_state != older.narrative_self_state
        assert newer.narrative_admission_checkpoint["stage"] == "S2"
        assert newer.last_processed_event_sequence > older.last_processed_event_sequence
        reviews = tuple(
            row
            for row in journal.inspect().records
            if row.event_type is AgentEventType.NARRATIVE_REVIEW
        )
        assert sum(row.lifecycle is EventLifecycle.COMPLETED for row in reviews) == 2
        high_water = journal.inspect().processing_high_water
        lost_source = client.app.state.experience_store.record_path(
            experience_id_for_event(refs[3].event_id, refs[3].event_sequence), 0
        )
        lost_source_bytes = lost_source.read_bytes()
        assert predecessor.read_bytes() == predecessor_bytes

    # Simulated process loss: invalidate only the successor's S2 transition,
    # plus a torn canonical snapshot. A valid current file would permit exact
    # current WAL repair instead of exercising true rollback.
    generation = wal.root / "generations" / f"{successor.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    transition = json.loads(lines[-1])
    assert transition["candidate_snapshot"]["narrative_self_state"]["revision"] == 2
    transition["record_hash"] = "0" * 64
    lines[-1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    corrupt_bytes = generation.read_bytes()
    store.path.write_bytes(b"{")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()
    assert wal.inspect_anchored_prefix()[1] == older

    observed: list[tuple[StateRecoveryResult, StartupReconciliationResult]] = []
    reconcile = StartupReconciliationCoordinator.reconcile_recovery_gate

    def observe_real_gate(
        coordinator: StartupReconciliationCoordinator,
        rolled_back: StateRecoveryResult,
    ) -> StartupReconciliationResult:
        assert rolled_back.true_rollback_performed
        assert rolled_back.external_reconciliation_required
        assert rolled_back.snapshot == older
        assert rolled_back.processing_high_water == high_water
        inspected = coordinator.journal.inspect()
        completed = inspected.records[-1]
        assert completed.lifecycle is EventLifecycle.RECOVERY_COMPLETED
        assert completed.recovery_category is EventRecoveryCategory.TRUE_ROLLBACK
        assert inspected.external_reconciliation_required
        manifest = coordinator.state_recovery.wal.inspect().active_manifest
        assert manifest is not None and manifest.external_reconciliation_required
        # Real mandatory gate rejects mutation BEFORE any publication.
        rejected = AgentEvent(
            second.event.event_id,
            AgentEventType.NARRATIVE_REVIEW,
            AgentEventSource.RUNTIME_NARRATIVE_REVIEW,
            second.event.occurred_at,
            high_water + 1,
        )
        before = coordinator.journal.path.read_bytes()
        with pytest.raises(StateRecoveryError, match="recovery-gated"):
            coordinator.state_recovery.commit_internal_candidate(rejected, older, newer)
        assert coordinator.journal.path.read_bytes() == before
        result = reconcile(coordinator, rolled_back)
        observed.append((rolled_back, result))
        return result

    restarted_app = create_app(settings)
    provider = SequencedProvider()
    restarted_app.state.model_provider = provider
    restarted_app.state.memory_system = DualMemorySystem(settings)
    restarted_app.state.adapter_registry = AdapterRegistry(settings)
    no_replay = Mock(side_effect=AssertionError("startup replayed cognition/admission"))
    with monkeypatch.context() as guard:
        guard.setattr(
            StartupReconciliationCoordinator,
            "reconcile_recovery_gate",
            observe_real_gate,
        )
        guard.setattr(SequencedProvider, "generate", no_replay)
        guard.setattr(PromptBuilder, "build", no_replay)
        guard.setattr(DualMemorySystem, "retrieve_context", no_replay)
        guard.setattr(WorkingMemory, "select", no_replay)
        guard.setattr(NarrativeSelfSystem, "_review_committed_turns", no_replay)
        guard.setattr(QualifiedNarrativeSourceGate, "qualify", no_replay)
        with TestClient(restarted_app):
            assert len(observed) == 1
            rolled_back, reconciled = observed[0]
            assert reconciled.participants_consistent
            assert not reconciled.recovery.external_reconciliation_required
            assert reconciled.recovery.snapshot == older
            state = restarted_app.state
            assert not state.external_reconciliation_required
            assert state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
            assert state.agent_state_store.load() == older
            assert state.agent_state_store.path.read_bytes() == older_bytes
            restored = state.r15_state_port._capture_for_agent_state()
            assert restored == technical
            assert restored.narrative.canonical_bytes() == narrative_bytes
            assert (
                canonical_json(
                    restored.continuation.canonical_value(restored.narrative)
                )
                == continuation_bytes
            )
            inspected = state.event_journal.inspect()
            assert inspected.processing_high_water == high_water
            assert (
                tuple(
                    row
                    for row in inspected.records
                    if row.event_type is AgentEventType.NARRATIVE_REVIEW
                )
                == reviews
            )
            assert inspected.completed_startup_reconciliations[-1].completed
            assert inspected.terminal_gate_clear is not None
            assert any(
                row.lifecycle is EventLifecycle.CLEAR_PREPARED
                for row in inspected.records
            )
            manifest = state.state_wal.inspect().active_manifest
            assert (
                manifest is not None and not manifest.external_reconciliation_required
            )
            assert predecessor.read_bytes() == predecessor_bytes
            assert generation.read_bytes() == corrupt_bytes
            assert no_replay.call_count == 0
            assert provider.calls == 0
            guard.undo()

            # Losing actual E2 evidence after proof-bound gate clear must not
            # regenerate S2 from retained source identifiers or the old S2 WAL.
            # Preserve the bytes outside the owner rather than deleting history.
            missing = tmp_path / "unavailable-e2.json"
            lost_source.rename(missing)
            with pytest.raises(AgentRuntimeExecutionError) as failure:
                _admit(state.agent_runtime, state.r15_state_port.narrative, refs)
            assert isinstance(failure.value.__cause__, QualifiedProposalUnavailable)
            assert state.r15_state_port._capture_for_agent_state() == technical
            assert state.agent_state_store.path.read_bytes() == older_bytes
            assert missing.read_bytes() == lost_source_bytes
            after = state.event_journal.inspect()
            assert (
                sum(
                    row.event_type is AgentEventType.NARRATIVE_REVIEW
                    and row.lifecycle is EventLifecycle.COMPLETED
                    for row in after.records
                )
                == 2
            )
            assert after.records[-1].lifecycle is EventLifecycle.FAILED
            assert after.records[-1].event_id == failure.value.event.event_id
            assert provider.calls == 0
            print(
                "U5_TRUE_ROLLBACK",
                {
                    "retained_generation": str(anchor.generation_id),
                    "corrupt_successor": str(successor.active_generation_id),
                    "recovery_generation": str(
                        rolled_back.manifest.active_generation_id
                    ),
                    "snapshot_sequence": older.last_processed_event_sequence,
                    "processing_high_water": high_water,
                    "source_refusal_sequence": failure.value.event.processing_sequence,
                    "snapshot_sha256": sha256(older_bytes).hexdigest(),
                    "narrative_sha256": sha256(narrative_bytes).hexdigest(),
                    "continuation_sha256": sha256(continuation_bytes).hexdigest(),
                    "first_sources": refs[:3],
                    "missing_e2": refs[3],
                    "gate": "required -> proof-bound clear -> missing-source refusal",
                },
            )


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
