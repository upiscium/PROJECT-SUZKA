"""Whole-R13 production ownership, publication, restart and privacy regressions.

Any private domain mutation in this suite is an injected test handler, not a
production producer. U6 intentionally adds no admission or outcome producer.
"""

from pathlib import Path

import pytest

from suzka.attention.common import AttentionTargetKind, canonical_json
from suzka.attention.policy import (
    ATTENTION_PROMPT_AUTHORITY_INTRO,
    ATTENTION_PROMPT_SECTION_HEADERS,
    AttentionPromptReason,
)
from suzka.config import Settings
from suzka.memory import DualMemorySystem
from suzka.motivation import (
    CommitmentSystem, GoalSystem, MotivationSystem,
    R13Reference, R13ReferenceKind, R13PromptView,
)
from suzka.motivation.system import MotivationEvidence, MotivationMutationEvidence
from suzka.motivation.goal_system import GoalMutationEvidence
from suzka.runtime import (
    AgentEventSource, AgentEventType, AgentRuntimeDurabilityError,
    AgentRuntimeExecutionError, AgentRuntimeStatus, AgentStateSaveError,
    AgentStateSaveStage, AgentStateSnapshotV7, AgentStateSnapshotV8,
    AgentStateSnapshotV9,
    AgentStateStore, SuzkaMainLoop,
)
from test_fastapi_backend import _client, _settings, ThinkingProvider, admin_headers
from r14_attention_fixtures import coherent_r13_systems


def _states(loop: SuzkaMainLoop) -> tuple:
    return (loop.export_motivation_state(), loop.export_goal_state(),
            loop.export_commitment_state())


def _seed_v8(settings: Settings) -> tuple:
    motivation, goal, commitment = coherent_r13_systems()
    loop = SuzkaMainLoop(
        settings, ThinkingProvider(), DualMemorySystem(settings),
        motivation_system=motivation, goal_system=goal, commitment_system=commitment,
    )
    store = AgentStateStore(settings.agent_state.path, settings.emotion.baseline_surprisal)
    store.configure_value_contract(
        tuple(seed.to_declaration() for seed in settings.values.seeds), ()
    )
    captured = store.capture(loop, 20)
    assert isinstance(captured, AgentStateSnapshotV9)
    retained_fields = captured.model_dump(mode="python")
    retained_fields.pop("attention_state")
    retained_fields["schema_version"] = 8
    store.save(AgentStateSnapshotV8.model_validate(retained_fields))
    return _states(loop), loop.r13_view()


def _inject_motivation(loop: SuzkaMainLoop, runtime: object) -> None:
    event = runtime.current_event()
    assert event is not None and event.processing_sequence is not None
    source = R13Reference(R13ReferenceKind.EMOTION, "emotion:injected-test")
    evidence = MotivationEvidence.from_emotion(
        source, valence=0.8, arousal=0.9,
        origin_refs=(R13Reference(R13ReferenceKind.EVENT, event.event_id),),
        source_event_id=event.event_id, source_event_sequence=event.processing_sequence,
        observed_at=event.requested_at,
    )
    loop._motivation_system.apply_evidence(
        evidence, MotivationMutationEvidence(
            event.event_id, event.processing_sequence, event.requested_at, (source,),
        ),
    )


def test_production_ports_upgrade_retained_v7_on_first_commit_without_feedback(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        loop = client.app.state.main_loop
        store = client.app.state.agent_state_store
        assert isinstance(store.load(), AgentStateSnapshotV7)
        assert isinstance(store.capture(loop, 0), AgentStateSnapshotV9)
        before = _states(loop)
        for message in ("adopt this user request", "accept this responsibility", "repeat request"):
            response = client.post("/api/chat", json={"message": message, "attachments": []})
            assert response.status_code == 200
            assert _states(loop) == before
            assert loop.r13_view() == R13PromptView()
            assert isinstance(store.load(), AgentStateSnapshotV9)
        debug = client.post(
            "/api/chat/debug", headers=admin_headers(),
            json={"message": "model thinks it wants more", "attachments": [], "debug": True},
        )
        assert debug.status_code == 200
        assert _states(loop) == before


def test_populated_v8_restarts_exactly_and_model_response_never_changes_r13(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    expected, expected_view = _seed_v8(settings)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("production/restart invoked an R13 mutation producer")

    for system, names in (
        (MotivationSystem, ("apply_evidence", "ingest_candidate", "propose_goal")),
        (GoalSystem, ("ingest_proposal", "adopt", "defer", "resume", "abandon")),
        (CommitmentSystem, ("ingest_proposal", "accept", "release", "renegotiate")),
    ):
        for name in names:
            monkeypatch.setattr(system, name, forbidden)

    with _client(tmp_path, settings=settings) as client:
        loop = client.app.state.main_loop
        assert _states(loop) == expected
        assert loop.r13_view() == expected_view
        response = client.post("/api/chat", json={"message": "new priority", "attachments": []})
        assert response.status_code == 200
        assert _states(loop) == expected
        assert loop.r13_view() == expected_view
    with _client(tmp_path, settings=settings) as restarted:
        assert _states(restarted.app.state.main_loop) == expected
        assert restarted.app.state.main_loop.r13_view() == expected_view


def test_standard_prompt_renders_only_minimal_current_r13_authority(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    expected, expected_view = _seed_v8(settings)
    with _client(tmp_path, settings=settings) as client:
        response = client.post(
            "/api/chat/debug", headers=admin_headers(),
            json={"message": "describe current state", "attachments": [], "debug": True},
        )
        assert response.status_code == 200
        prompt = response.json()["prompt"]
        loop = client.app.state.main_loop
        attention_view = loop.attention_view()
        attention_snapshot = loop._attention_system.snapshot()
        assert attention_view.prompt is not None
        assert attention_view.revision == attention_snapshot.revision
        assert attention_view.event == attention_snapshot.last_event
        assert ATTENTION_PROMPT_AUTHORITY_INTRO in prompt
        assert prompt.count(ATTENTION_PROMPT_AUTHORITY_INTRO) == 1
        header_positions = tuple(
            prompt.index(header) for _, header in ATTENTION_PROMPT_SECTION_HEADERS
        )
        assert header_positions == tuple(sorted(header_positions))
        assert prompt.index(ATTENTION_PROMPT_AUTHORITY_INTRO) < header_positions[0]
        assert all(
            prompt.count(header) == 1
            for _, header in ATTENTION_PROMPT_SECTION_HEADERS
        )

        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in attention_snapshot.candidates
        }
        included_candidate_ids = set(attention_view.prompt.included_candidate_ids)
        assert included_candidate_ids <= set(candidates_by_id)
        assert included_candidate_ids
        included_decision_ids = {
            decision.candidate_id
            for decision in attention_view.prompt.decisions
            if decision.reason is AttentionPromptReason.INCLUDED
        }
        assert included_candidate_ids == included_decision_ids
        expected_rows = expected_view.canonical_value()
        row_ids_by_kind = {
            AttentionTargetKind.MOTIVATION: ("motivations", "motivation_id"),
            AttentionTargetKind.GOAL: ("goals", "goal_id"),
            AttentionTargetKind.COMMITMENT: ("commitments", "commitment_id"),
        }
        source_rows_by_kind: dict[AttentionTargetKind, dict[str, str]] = {}
        for kind, (section, identity_field) in row_ids_by_kind.items():
            source_rows = expected_rows[section]
            assert type(source_rows) is list
            rows_by_reference: dict[str, str] = {}
            for source_row in source_rows:
                assert type(source_row) is dict
                reference = source_row[identity_field]
                assert type(reference) is str
                rows_by_reference[reference] = canonical_json(source_row).decode(
                    "ascii"
                )
            source_rows_by_kind[kind] = rows_by_reference
        expected_render_order = [
            source_rows_by_kind[kind][
                candidates_by_id[candidate_id].target.reference
            ]
            for kind, _header in ATTENTION_PROMPT_SECTION_HEADERS
            if kind in source_rows_by_kind
            for candidate_id in attention_view.prompt.included_candidate_ids
            if candidates_by_id[candidate_id].target.kind is kind
        ]
        known_rows = {
            row
            for rows in source_rows_by_kind.values()
            for row in rows.values()
        }
        actual_render_order = [
            line for line in prompt.splitlines() if line in known_rows
        ]
        assert actual_render_order == expected_render_order
        assert loop.r13_view() == expected_view
        for snapshot in expected:
            for receipt in snapshot.event_receipts:
                assert receipt.event_id not in prompt
                assert receipt.receipt_digest not in prompt
        for record in (*expected[1].records, *expected[2].records):
            assert record.subject_admission is not None
            assert record.subject_admission.admission_digest not in prompt
            for proof in record.subject_transition_proofs:
                assert proof.transition_digest not in prompt
        assert _states(client.app.state.main_loop) == expected


@pytest.mark.parametrize("failed_domain", ["handler", "goal_ingestion"])
def test_failed_injected_domain_operation_rolls_back_without_speculative_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_domain: str
) -> None:
    with _client(tmp_path) as client:
        loop, runtime = client.app.state.main_loop, client.app.state.agent_runtime
        before, committed_view = _states(loop), loop.r13_view()
        canonical = client.app.state.agent_state_store.path.read_bytes()
        if failed_domain == "goal_ingestion":
            def reject_ingestion(*_args: object) -> None:
                raise ValueError("injected Goal ingestion failure")
            monkeypatch.setattr(loop._goal_system, "ingest_proposal", reject_ingestion)

        def handler() -> None:
            _inject_motivation(loop, runtime)
            assert loop.export_motivation_state() != before[0]
            assert loop.r13_view() == committed_view
            if failed_domain == "goal_ingestion":
                proposal = loop._motivation_system.propose_goal(
                    loop.export_motivation_state().records[0].motivation_id
                )
                assert proposal is not None
                event = runtime.current_event()
                assert event is not None and event.processing_sequence is not None
                loop._goal_system.ingest_proposal(
                    proposal, GoalMutationEvidence(
                        event.event_id, event.processing_sequence,
                        event.requested_at, proposal.evidence_refs,
                    ),
                )
            raise ValueError("injected handler failure")

        with pytest.raises(AgentRuntimeExecutionError):
            runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, handler).result(10)
        assert _states(loop) == before
        assert loop.r13_view() == committed_view
        assert client.app.state.agent_state_store.path.read_bytes() == canonical
        assert runtime.status is AgentRuntimeStatus.ACCEPTING


def test_failed_v8_publication_keeps_committed_r13_view_and_restart_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        loop, runtime = client.app.state.main_loop, client.app.state.agent_runtime
        store = client.app.state.agent_state_store
        committed_view = loop.r13_view()
        before = _states(loop)
        canonical = store.path.read_bytes()

        def reject_publication(_snapshot: object) -> None:
            raise AgentStateSaveError(AgentStateSaveStage.ATOMIC_REPLACE, published=False)

        monkeypatch.setattr(store, "save", reject_publication)
        with pytest.raises(AgentRuntimeDurabilityError):
            runtime.submit(
                AgentEventType.CHAT, AgentEventSource.API_CHAT,
                lambda: _inject_motivation(loop, runtime),
            ).result(10)
        assert loop.r13_view() == committed_view
        assert store.path.read_bytes() == canonical
        assert runtime.status is AgentRuntimeStatus.FAILED
    with _client(tmp_path, settings=settings) as restarted:
        assert _states(restarted.app.state.main_loop) == before
        assert restarted.app.state.main_loop.r13_view() == committed_view


def test_projection_preflight_failure_precedes_durable_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        loop, runtime = client.app.state.main_loop, client.app.state.agent_runtime
        before = loop.r13_view()
        store, wal, journal = (client.app.state.agent_state_store,
                               client.app.state.state_wal, client.app.state.event_journal)
        canonical, wal_before = store.path.read_bytes(), wal.inspect()

        def reject_view(_snapshot: object) -> None:
            raise ValueError("injected projection preflight failure")

        monkeypatch.setattr(loop, "_prepare_committed_r13_view", reject_view)
        with pytest.raises(AgentRuntimeDurabilityError):
            runtime.submit(
                AgentEventType.CHAT, AgentEventSource.API_CHAT,
                lambda: _inject_motivation(loop, runtime),
            ).result(10)
        assert loop.r13_view() == before
        assert store.path.read_bytes() == canonical
        assert wal.inspect() == wal_before
        assert not any(record.lifecycle.value == "prepared" for record in journal.records)


def test_published_v8_failure_fail_stops_and_restart_recovers_exact_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        loop, runtime = client.app.state.main_loop, client.app.state.agent_runtime
        store = client.app.state.agent_state_store
        committed_view = loop.r13_view()
        canonical = store.path.read_bytes()
        save = store.save

        def publish_then_fail(snapshot: object) -> None:
            save(snapshot)
            raise AgentStateSaveError(
                AgentStateSaveStage.PARENT_FSYNC, published=True
            )

        monkeypatch.setattr(store, "save", publish_then_fail)
        with pytest.raises(AgentRuntimeDurabilityError):
            runtime.submit(
                AgentEventType.CHAT, AgentEventSource.API_CHAT,
                lambda: _inject_motivation(loop, runtime),
            ).result(10)

        assert runtime.status is AgentRuntimeStatus.FAILED
        assert loop.r13_view() == committed_view
        assert store.path.read_bytes() != canonical
        expected = _states(loop)
        persisted = store.load()
        assert isinstance(persisted, AgentStateSnapshotV9)
        assert persisted.r13_state.motivation.restore() == expected[0]
        expected_view = R13PromptView.from_snapshots(*expected)
        assert expected_view != committed_view

    def forbid_replay(*_args: object, **_kwargs: object) -> None:
        pytest.fail("restart replayed Motivation policy")

    monkeypatch.setattr(MotivationSystem, "apply_evidence", forbid_replay)
    with _client(tmp_path, settings=settings) as restarted:
        assert _states(restarted.app.state.main_loop) == expected
        assert restarted.app.state.main_loop.r13_view() == expected_view


def test_successful_injected_commit_publishes_one_bundle_and_retry_is_idempotent(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        loop, runtime = client.app.state.main_loop, client.app.state.agent_runtime
        before_view = loop.r13_view()
        runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT,
            lambda: _inject_motivation(loop, runtime),
        ).result(10)
        snapshot = loop.export_motivation_state()
        assert loop.r13_view() != before_view
        assert len(loop.r13_view().motivations) == 1
        entry = snapshot.evidence_ledger[0]
        receipt = snapshot.event_receipts[0]
        loop._motivation_system.apply_evidence(
            entry.evidence,
            MotivationMutationEvidence(receipt.event_id, receipt.event_sequence,
                                       receipt.recorded_at, receipt.evidence_refs),
        )
        assert loop.export_motivation_state() == snapshot
        assert client.app.state.agent_state_store.load().r13_state.motivation.restore() == snapshot


def test_production_mainloop_exposes_no_mutable_r13_read_or_later_producer(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        loop = client.app.state.main_loop
        for name in ("motivation_system", "goal_system", "commitment_system", "adopt_goal",
                     "accept_commitment", "complete_goal", "fail_goal", "fulfill_commitment",
                     "breach_commitment", "r13_timer", "r13_scheduler"):
            assert not hasattr(loop, name)
        assert type(loop.r13_view()) is R13PromptView
